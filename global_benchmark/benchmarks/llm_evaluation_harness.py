import subprocess
from global_benchmark.benchmarks import BenchmarkRunner
from typing import Union
from global_benchmark.utils import RunInfo, Task, SlurmInfo, BenchmarkFramework
from rich import print
import os


class LlmEvaluationHarnessRunner(BenchmarkRunner):
    def __init__(
        self,
        run_id: str,
        run_info: RunInfo,
        slurm_info: Union[None, SlurmInfo],
        images_directory: str,
    ):
        super().__init__(
            run_id,
            BenchmarkFramework.LLM_EVALUATION_HARNESS,
            run_info,
            slurm_info,
            images_directory,
        )

    def run(self, model: str, task: Task, slurm: bool = False):
        """
        Run the benchmark.

        Args:
            model (str): The model to run the benchmark on.
            task (Task): The task to execute.
        """

        self.exec(
            (
                "lm_eval "
                "--model vllm "
                f"--model_args pretrained={model},tensor_parallel_size={self.run_info['tensor_parallel_size']},dtype={self.run_info['dtype']},gpu_memory_utilization=0.8 "
                f"--tasks {task} "
                "--batch_size auto "
                f"--output_path ./results/temp/{self.run_id}/{task} "
                "--confirm_run_unsafe_code "
            ),
            slurm,
        )

    def exec(
        self,
        command: str,
        slurm: bool,
    ):
        """
        Execute a command in the container.
        """

        cmd = (
            "apptainer "
            "exec "
            "--nv "
            "-B /scratch "
            f"{self.images_directory}/{self.framework} "
            f"{command}"
        )

        if slurm:
            cmd = self.get_slurm_command(cmd)

        print(f"[black]Running command: {cmd}[black]")
        return subprocess.run(
            cmd,
            check=True,
            shell=True,
            env=os.environ.copy(),
        )
