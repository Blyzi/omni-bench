from abc import ABCMeta, abstractmethod
from pathlib import Path
import subprocess
from typing import Any, List, Literal, Union
import os
import json
from omni.utils.enums import Benchmark, Task
from omni.utils.schemas import RunConfig, SlurmConfig, ApptainerBind


class BenchmarkRunner(metaclass=ABCMeta):
    needs_parameters = False

    def __init__(
        self,
        run_id: str,
        framework: Benchmark,
        run_config: RunConfig,
        slurm_config: Union[None, SlurmConfig],
    ):
        self.run_id = run_id
        self.framework = framework
        self.run_config = run_config
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

        cmd = (
            "sbatch "
            '--job-name="omni" '
            f"--mem={self.slurm_config.mem} "
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
                f"-p {self.slurm_config.cpu_partition} "
                f"-c {self.slurm_config.cpus_per_task} "
            )

        elif slurm_partition == "gpu":
            cmd += (
                f"-p {self.slurm_config.gpu_partition} "
                f"--gres={self.slurm_config.gpu_gres} "
                f"--cpus-per-gpu={self.slurm_config.cpus_per_gpu} "
            )

            if self.slurm_config.gpu_constraint:
                cmd += f'--constraint="{self.slurm_config.gpu_constraint}" '

        if self.slurm_config.account:
            cmd += f"-A {self.slurm_config.account} "

        if self.slurm_config.exclude:
            cmd += f'--exclude="{self.slurm_config.exclude}" '

        if dependencies:
            cmd += f" --dependency={':'.join(dependencies)} "

        return cmd

    def get_apptainer_command(
        self,
        command: str,
        image: Path,
        binds: list[ApptainerBind] = [],
        cwd: Path = Path("."),
    ) -> str:
        """
        Get the Apptainer command for the benchmark.
        """

        cmd = (
            "apptainer "
            "exec "
            "--nv "
            "-c "
            f"--cwd {cwd} "
            f"-B {','.join((bind.source.as_posix() + ':' + bind.target.as_posix() for bind in self.run_config.binds + binds))} "
            f"{image} "
            f"{command} "
        )

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
        apptainer: bool,
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
