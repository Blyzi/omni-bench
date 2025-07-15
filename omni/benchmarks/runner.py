from abc import ABCMeta, abstractmethod
from pathlib import Path
import subprocess
from typing import Any, List, Literal, Union
import os
import json
from omni.utils.enums import Benchmark, Task
from omni.utils.schemas import RunConfig, SlurmConfig, ContainerBind, BenchmarkConfig
from rich import print


class BenchmarkRunner(metaclass=ABCMeta):
    needs_parameters = False

    def __init__(
        self,
        run_id: str,
        framework: Benchmark,
        run_config: RunConfig,
        benchmark_config: BenchmarkConfig,
        slurm_config: Union[None, SlurmConfig],
    ):
        self.run_id = run_id
        self.framework = framework
        self.run_config = run_config
        self.benchmark_config = benchmark_config
        self.slurm_config = slurm_config

    def get_slurm_command(
        self,
        command: str,
        slurm_partition: Union[Literal["cpu"], Literal["gpu"]],
        slurm_dependency: List[str] = [],
    ) -> str:
        """
        Get the SLURM command for the benchmark.
        """
        if not self.slurm_config:
            raise ValueError("SLURM information is not provided.")

        if self.slurm_config.prescript:
            command = f"{self.slurm_config.prescript} && {command}"

        cmd = (
            "sbatch "
            '--job-name="omni" '
            "--output=./logs/slurm-%j.out "
            "--error=./logs/slurm-%j.err "
            "--kill-on-invalid-dep=yes "
            f'--wrap="{command}" '
        )

        dependencies = (
            ["afterok:" + ":".join(slurm_dependency)] if slurm_dependency else []
        )

        if slurm_partition == "cpu":
            cmd += (
                " ".join(
                    f"--{key.replace('_', '-')}='{value}'"
                    for key, value in self.slurm_config.cpu.model_dump().items()
                )
                + " "
            )

        elif slurm_partition == "gpu":
            cmd += (
                " ".join(
                    f"--{key.replace('_', '-')}='{value}'"
                    for key, value in self.slurm_config.gpu.model_dump().items()
                )
                + " "
            )

        if dependencies:
            cmd += f" --dependency={':'.join(dependencies)} "

        return cmd

    def get_container_command(
        self,
        command: str,
        image_name: str,
        binds: list[ContainerBind] = [],
        cwd: Path = Path("."),
    ) -> str:
        """
        Get the Container command for the benchmark.
        """

        cmd = f"{self.run_config.container_system} exec --nv -c --cwd {cwd} "

        if len(self.run_config.binds + binds) > 0:
            cmd += f"-B {','.join((bind.source.as_posix() + ':' + bind.target.as_posix() for bind in self.run_config.binds + binds))} "

        cmd += f"{self.run_config.images_directory}/{image_name}.sif {command} "

        return cmd

    def exec(
        self,
        command: str,
        slurm: bool,
    ) -> str:
        """
        Execute a command in the container.
        """

        print(f"[black]Running command: {command}[/black]")
        output = subprocess.run(
            command,
            shell=True,
            check=True,
            text=slurm,
            env=os.environ.copy(),
            stdout=subprocess.PIPE if slurm else None,
        )

        job_id = output.stdout.strip().split()[-1] if slurm else ""

        return job_id

    @abstractmethod
    def run(self, model: str, task: Task, slurm: bool) -> None:
        """
        Run the benchmark.

        Args:
            model (str): The model to run the benchmark on.
            task (Task): The task to execute.
            slurm (bool): Whether to run the benchmark with SLURM.
        """
        pass

    @abstractmethod
    def command_wrapper(
        self,
        command: str,
        container: bool,
        slurm: bool,
        slurm_partition: Union[Literal["cpu"], Literal["gpu"]],
        slurm_dependency: List[str] = [],
    ) -> str:
        """
        Execute a command in the container.
        """
        pass

    def store(
        self,
        model: str,
        task: Task,
        result: dict[str, Any],
    ) -> None:
        """
        Save the results of the benchmark.
        """

        filename = f"results/{self.run_id}.json"
        data: dict[str, Union[str, list]]

        if os.path.exists(filename):
            with open(filename, "r") as file:
                data = json.load(file)
        else:
            data = {
                "model": model,
            }

        if task not in data:
            data[task] = [result]
        elif isinstance(data[task], list):
            data[task].append(result)  # type: ignore data[task] is a list

        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
