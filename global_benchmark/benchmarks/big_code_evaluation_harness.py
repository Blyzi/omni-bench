import ast
import subprocess
import inquirer
import typer
from global_benchmark.benchmarks import BenchmarkRunner
from global_benchmark.utils import BenchmarkFramework, Task, RunInfo, SlurmInfo
from rich import print


class BigCodeEvaluationHarnessRunner(BenchmarkRunner):
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

    def run(
        self,
        model: str,
        task: Task,
        slurm: bool,
    ):
        """
        Run the benchmark.
        """

        for temperature, n_samples in self.parameters:
            if task == Task.MULTIPLE:
                for subtask in [
                    "multiple-cljcpp",
                    "multiple-cs",
                    "multiple-d",
                    "multiple-dart",
                    "multiple-elixir",
                    "multiple-go",
                    "multiple-hs",
                    "multiple-java",
                    "multiple-jl",
                    "multiple-js",
                    "multiple-lua",
                    "multiple-mlpl",
                    "multiple-php",
                    "multiple-py",
                    "multiple-r",
                    "multiple-rb",
                    "multiple-rkt",
                    "multiple-rs",
                    "multiple-scala",
                    "multiple-sh",
                    "multiple-swift",
                    "multiple-ts",
                ]:
                    self.exec(
                        (
                            "accelerate launch main.py "
                            f"--model {model} "
                            f"--tasks {subtask} "
                            f"--temperature {temperature} "
                            f"--n_samples {n_samples} "
                            f"--precision {self.run_info['dtype']} "
                            "--allow_code_execution "
                            f"--metric_output_path /results/temp/{self.run_id}/{subtask}.json "
                        ),
                        slurm,
                    )

            else:
                self.exec(
                    (
                        "accelerate launch main.py "
                        f"--model {model} "
                        f"--tasks {task} "
                        f"--temperature {temperature} "
                        f"--n_samples {n_samples} "
                        f"--precision {self.run_info['dtype']} "
                        "--allow_code_execution "
                        f"--metric_output_path /results/temp/{self.run_id}/{task}.json "
                    ),
                    slurm,
                )

    def exec(self, command: str, slurm: bool) -> subprocess.CompletedProcess:
        """
        Execute a command in the container.
        """

        cmd = (
            "apptainer "
            "exec "
            "--nv "
            "-B ./results:/results,/scratch "
            "--cwd /bigcode-evaluation-harness "
            f"{self.images_directory}/{self.framework} "
            f"{command}"
        )

        if slurm:
            cmd = self.get_slurm_command(cmd)

        print(f"[blue]Executing command: {cmd}[/blue]")
        return subprocess.run(cmd, shell=True, check=True)
