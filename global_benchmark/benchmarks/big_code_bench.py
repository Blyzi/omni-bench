import os
import subprocess
import typer
from global_benchmark.benchmarks import BenchmarkRunner
import inquirer
import ast
from global_benchmark.utils import BenchmarkFramework, RunInfo, Task, SlurmInfo
from rich import print


class BigCodeBenchRunner(BenchmarkRunner):
    needs_parameters = True

    def __init__(
        self,
        run_id: str,
        run_info: RunInfo,
        slurm_info: SlurmInfo,
        images_directory: str,
    ):
        super().__init__(
            run_id,
            BenchmarkFramework.BIG_CODE_BENCHMARK,
            run_info,
            slurm_info,
            images_directory,
        )

        print(
            "[yellow bold]WARNING: The benchmark BigCodeBench force the bfloat16 precision, meaning that your GPU must support it or the run will fails[/yellow bold]"
        )

    def get_parameters(self):
        """
        Get the parameters for the benchmark.
        """

        response = inquirer.prompt(
            [
                inquirer.Text(
                    "parameters",
                    message="Enter the benchmark parameters like this: [(temperature, n_samples)]",
                ),
            ]
        )

        if response is None:
            raise typer.BadParameter(
                "No parameters provided. Please provide the parameters in the correct format."
            )

        self.parameters = ast.literal_eval(response["parameters"])

    def run(self, model: str, task: Task, slurm: bool = False):
        """
        Run the benchmark on the BigCode dataset.
        """
        for temperature, n_samples in self.parameters:
            job_res = self.exec(
                (
                    "bigcodebench.generate "
                    f"--model {model} "
                    f"--temperature {temperature} "
                    f"--n_samples {n_samples} "
                    f"--split {'complete' if task == Task.BIG_CODE_BENCHMARK_COMPLETE else 'instruct'} "
                    "--subset full "
                    f"--tp {self.run_info['tensor_parallel_size']} "
                    "--backend vllm "
                    f"--root ./results/temp/{self.run_id}/{task}_{temperature}_{n_samples} "
                    f"&& cp -r ./results/temp/{self.run_id}/{task}_{temperature}_{n_samples} ./results/temp/{self.run_id}/{task}_hard_{temperature}_{n_samples} "
                ),
                slurm,
            )

            self.exec(
                (
                    "bigcodebench.evaluate "
                    f"--model {model} "
                    f"--temperature {temperature} "
                    f"--n_samples {n_samples} "
                    f"--samples results/temp/{self.run_id}/{task}_{temperature}_{n_samples}/{model.replace('/', '--')}--main--bigcodebench-complete--vllm-{temperature}-{n_samples}-sanitized_calibrated.jsonl "
                    "--execution local "
                    f"--split {'complete' if task == Task.BIG_CODE_BENCHMARK_COMPLETE else 'instruct'} "
                    "--subset full "
                    "--backend vllm "
                ),
                slurm,
                eval=True,
                slurm_dependency=job_res.stdout.strip().split(" ")[-1],
            )

            self.exec(
                (
                    "bigcodebench.evaluate "
                    f"--model {model} "
                    f"--temperature {temperature} "
                    f"--n_samples {n_samples} "
                    f"--samples results/temp/{self.run_id}/{task}_hard_{temperature}_{n_samples}/{model.replace('/', '--')}--main--bigcodebench-complete--vllm-{temperature}-{n_samples}-sanitized_calibrated.jsonl "
                    "--execution local "
                    f"--split {'complete' if task == Task.BIG_CODE_BENCHMARK_COMPLETE else 'instruct'} "
                    "--subset hard "
                    "--backend vllm "
                ),
                slurm,
                eval=True,
                slurm_dependency=job_res.stdout.strip().split(" ")[-1],
            )

    def exec(
        self, command: str, slurm: bool, eval: bool = False, slurm_dependency: str = ""
    ):
        """
        Execute a command in the container.
        """
        # Implement the logic to execute a command in the BigCode evaluation harness container
        cmd = (
            "apptainer "
            "exec "
            "--nv "
            "-B /scratch "
            f"{self.images_directory}/{self.framework}{'_eval' if eval else '_gen'} "
            f"{command}"
        )

        if slurm:
            cmd = self.get_slurm_command(cmd, slurm_dependency)

        print(f"[black]Running command: {cmd}[black]")
        return subprocess.run(
            cmd,
            shell=True,
            check=True,
            env=os.environ.copy(),
            stdout=subprocess.PIPE if slurm else None,
            text=slurm,
        )
