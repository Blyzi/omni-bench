import json
from pathlib import Path
from typing import List, Literal, Union
import inquirer
import typer
from omni.benchmarks.runner import BenchmarkRunner
from omni.utils.enums import Benchmark, Task
from omni.utils.schemas import RunConfig, BenchmarkConfig, SlurmConfig, ContainerBind


class RulerRunner(BenchmarkRunner):
    needs_parameters = True

    def __init__(
        self,
        run_id: str,
        run_config: RunConfig,
        benchmark_config: BenchmarkConfig,
        slurm_config: Union[None, SlurmConfig],
    ):
        super().__init__(
            run_id,
            Benchmark.RULER,
            run_config,
            benchmark_config,
            slurm_config,
        )

    def get_parameters(self):
        """
        Get the parameters for the benchmark.
        """

        response = inquirer.prompt(
            [
                inquirer.List(
                    "prompt_template",
                    choices=[
                        "meta-chat",
                        "meta-llama3",
                        "base",
                        "jamba",
                    ],
                    message="Choose a prompt template",
                ),
                inquirer.List(
                    "context_length",
                    choices=["131072", "65536", "32768", "16384", "8192", "4096"],
                    message="Choose a context length",
                ),
            ]
        )

        if (
            response is None
            or not response.get("prompt_template")
            or not response.get("context_length")
        ):
            raise typer.BadParameter(
                "No parameters provided. Please provide the parameters in the correct format."
            )

        self.parameters = {
            "prompt_template": response["prompt_template"],
            "context_length": int(response["context_length"]),
        }

    def run(self, model: str, task: Task, slurm: bool = False):
        """
        Run the benchmark.

        Args:
            model (str): The model to run the benchmark on.
            task (Task): The task to execute.
        """

        mkdir_job = self.exec(
            self.command_wrapper(
                f"mkdir -p ./results/temp/{self.run_id}/{Task.RULER_SYNTHETIC}",
                container=False,
                slurm=slurm,
                slurm_compute="cpu",
            ),
            slurm=slurm,
        )

        # Generate context lengths based on the provided context length
        start = 4096
        context_lengths = []
        while start <= self.parameters["context_length"]:
            context_lengths.append(str(start))
            start *= 2

        ruler_job = self.exec(
            self.command_wrapper(
                (
                    "bash -c '"
                    f"export MODEL_TEMPLATE_TYPE={self.parameters['prompt_template']} && "
                    f"export MODEL_PATH={model} && "
                    f"export MODEL_FRAMEWORK=vllm && "
                    f"export GPUS={self.run_config.tensor_parallel_size} && "
                    f"export THREADS={self.benchmark_config.ruler.thread_count} && "
                    f"export SEQ_LENGTHS='{' '.join(context_lengths)}' && "
                    f"./run.sh {model} synthetic' "
                ),
                container=True,
                slurm=slurm,
                slurm_compute="gpu",
                slurm_dependency=[mkdir_job],
            ),
            slurm=slurm,
        )

        self.exec(
            self.command_wrapper(
                f"uv run omni save {self.run_id} {task} {model}",
                container=False,
                slurm=slurm,
                slurm_compute="cpu",
                slurm_dependency=[ruler_job],
            ),
            slurm=slurm,
        )

    def command_wrapper(
        self,
        command: str,
        container: bool,
        slurm: bool,
        slurm_compute: Union[Literal["cpu"], Literal["gpu"]],
        slurm_dependency: List[str] = [],
    ):
        """
        Execute a command in the container.
        """

        if container:
            command = self.get_container_command(
                command,
                self.framework.value,
                [
                    ContainerBind(
                        source=Path(
                            f"./results/temp/{self.run_id}/{Task.RULER_SYNTHETIC}"
                        ),
                        target=Path("/RULER/scripts/benchmark_root"),
                    )
                ],
                Path("/RULER/scripts"),
            )

        if slurm:
            command = self.get_slurm_command(command, slurm_compute, slurm_dependency)

        return command

    def save(self, model: str, task: Task):
        """
        Save the results of the benchmark.
        TODO: Implement the save logic for RulerRunner
        """

    pass
