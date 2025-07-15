import ast
import json
from pathlib import Path
from typing import List, Literal, Union
import inquirer
import typer
from omni.benchmarks import BenchmarkRunner
from omni.utils.enums import Benchmark, Task
from omni.utils.schemas import ApptainerBind, SlurmConfig, RunConfig, BenchmarkConfig
from omni.utils.functions import get_short_precision


class BigCodeEvaluationHarnessRunner(BenchmarkRunner):
    needs_parameters = True

    def __init__(
        self,
        run_id: str,
        run_config: RunConfig,
        benchmark_config: BenchmarkConfig,
        slurm_config: SlurmConfig,
    ):
        super().__init__(
            run_id,
            Benchmark.BIG_CODE_EVALUATION_HARNESS,
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

        jobs = []

        for temperature, n_samples in self.parameters:
            create_folder_job = self.exec(
                self.command_wrapper(
                    f"mkdir -p ./results/temp/{self.run_id}/{task} ",
                    apptainer=False,
                    slurm=slurm,
                    slurm_partition="cpu",
                ),
                slurm=slurm,
            )

            cmd = (
                "accelerate launch main.py "
                f"--model {model} "
                f"--tasks {task} "
                f"--precision {get_short_precision(self.run_config.dtype)} "
                "--allow_code_execution "
                f"--metric_output_path /results/temp/{self.run_id}/{task}/{task}_{temperature}_{n_samples}.json "
                f"--temperature {temperature} "
                f"--n_samples {n_samples} "
            )

            if temperature == 0:
                cmd += "--do_sample=False "

            benchmark_job = self.exec(
                self.command_wrapper(
                    cmd,
                    apptainer=True,
                    slurm=slurm,
                    slurm_dependency=[create_folder_job],
                    slurm_partition="gpu",
                ),
                slurm=slurm,
            )

            jobs.append(benchmark_job)

        self.exec(
            self.command_wrapper(
                f"uv run omni save {self.run_id} {task} {model}",
                apptainer=False,
                slurm=slurm,
                slurm_partition="cpu",
                slurm_dependency=jobs,
            ),
            slurm=slurm,
        )

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

        if apptainer:
            command = self.get_apptainer_command(
                command,
                self.run_config.images_directory / self.framework.value,
                [ApptainerBind(source=Path("results"), target=Path("/results"))],
                Path("/bigcode-evaluation-harness"),
            )

        if slurm:
            command = self.get_slurm_command(command, slurm_partition, slurm_dependency)

        return command

    def save(
        self,
        model: str,
        task: Task,
    ) -> None:
        """
        Save the results of the benchmark.
        """

        files = Path(f"results/temp/{self.run_id}/{task}").rglob("*.json")

        for filename in files:
            with open(filename, "r") as f:
                data = json.load(f)
                self.store(
                    model,
                    task,
                    {
                        **data[task],
                        "temperature": data["config"]["temperature"],
                        "n_samples": data["config"]["n_samples"],
                    },
                )
