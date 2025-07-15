import json
from omni.benchmarks import BenchmarkRunner
from typing import List, Literal, Union
from omni.utils.enums import Benchmark, Task
from omni.utils.schemas import RunConfig, SlurmConfig, ContainerBind
from pathlib import Path
from omni.utils.schemas import BenchmarkConfig


class LlmEvaluationHarnessRunner(BenchmarkRunner):
    def __init__(
        self,
        run_id: str,
        run_config: RunConfig,
        benchmark_config: BenchmarkConfig,
        slurm_config: Union[None, SlurmConfig],
    ):
        super().__init__(
            run_id,
            Benchmark.LLM_EVALUATION_HARNESS,
            run_config,
            benchmark_config,
            slurm_config,
        )

    def run(self, model: str, task: Task, slurm: bool = False):
        """
        Run the benchmark.

        Args:
            model (str): The model to run the benchmark on.
            task (Task): The task to execute.
        """

        job = self.exec(
            self.command_wrapper(
                (
                    "lm_eval "
                    "--model vllm "
                    f"--model_args pretrained={model},tensor_parallel_size={self.run_config.tensor_parallel_size},dtype={self.run_config.dtype},gpu_memory_utilization=0.9,{','.join(f'{key}={value}' for key, value in self.benchmark_config.llm_evaluation_harness.model_dump().items())} "
                    f"--tasks {task} "
                    "--batch_size auto "
                    f"--output_path /results/temp/{self.run_id}/{task} "
                    f"--log_samples "
                    "--confirm_run_unsafe_code "
                ),
                container=True,
                slurm=slurm,
                slurm_compute="gpu",
            ),
            slurm=slurm,
        )

        self.exec(
            self.command_wrapper(
                f"uv run omni save {self.run_id} {task} {model}",
                container=False,
                slurm=slurm,
                slurm_compute="cpu",
                slurm_dependency=[job],
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
                [ContainerBind(source=Path("results"), target=Path("/results"))],
            )

        if slurm:
            command = self.get_slurm_command(command, slurm_compute, slurm_dependency)

        return command

    def save(self, model: str, task: Task):
        """
        Save the results of the benchmark.
        """

        filename = next(
            Path(f"results/temp/{self.run_id}/{task}").rglob("*.json"), None
        )

        if filename is not None:
            with open(filename, "r") as f:
                result = json.load(f)
                self.store(model, task, result["results"][task])
