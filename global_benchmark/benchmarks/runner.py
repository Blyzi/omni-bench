from abc import ABCMeta, abstractmethod
from subprocess import CompletedProcess
from typing import Union
from global_benchmark.utils import BenchmarkFramework, RunInfo, Task, SlurmInfo


class BenchmarkRunner(metaclass=ABCMeta):
    needs_parameters = False

    def __init__(
        self,
        run_id: str,
        framework: BenchmarkFramework,
        run_info: RunInfo,
        slurm_info: Union[None, SlurmInfo],
        images_directory: str,
    ):
        self.run_id = run_id
        self.framework = framework
        self.run_info = run_info
        self.slurm_info = slurm_info
        self.images_directory = images_directory

    def get_slurm_command(
        self,
        command: str,
        slurm_dependency: str = "",
    ) -> str:
        """
        Get the SLURM command for the benchmark.
        """
        if not self.slurm_info:
            raise ValueError("SLURM information is not provided.")

        cmd = (
            "sbatch "
            f"-p {self.slurm_info['partition']} "
            f"--gres={self.slurm_info['gres']} "
            f"--cpus-per-gpu={self.slurm_info['cpus_per_gpu']} "
            f"--mem={self.slurm_info['mem']} "
            f'--job-name="codebench" '
            "--output=./logs/slurm-%j.out "
            "--error=./logs/slurm-%j.err "
            "--kill-on-invalid-dep=yes "
            f'--wrap="{command}" '
        )

        if self.slurm_info["account"]:
            cmd += f"-A {self.slurm_info['account']} "

        if self.slurm_info["constraint"]:
            cmd += f'--constraint="{self.slurm_info["constraint"]}" '

        if slurm_dependency:
            cmd += f'--dependency="afterok:{slurm_dependency}" '

        return cmd

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
    def exec(self, command: str, slurm: bool) -> CompletedProcess:
        """
        Execute a command in the container.
        """
        pass
