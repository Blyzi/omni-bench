from typing import List, Union
import typer
import inquirer
from typing_extensions import Annotated
from global_benchmark.utils import SlurmInfo, Task, BenchmarkFramework
from global_benchmark import services
import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

app = typer.Typer()


@app.command()
def run(
    model: str = "",
    tasks: Annotated[List[Task], typer.Option()] = [],
    dtype: str = "auto",
    tensor_parallel_size: int = 1,
    images_directory: str = "./images",
    slurm: bool = False,
    slurm_partition: str = "",
    slurm_gres: str = "",
    slurm_cpus_per_gpu: int = 0,
    slurm_mem: str = "",
    slurm_account: str = "",
    slurm_constraint: str = "",
):
    """
    Run the benchmark.
    """

    if model == "":
        response = inquirer.prompt(
            [
                inquirer.Text("model", message="Enter the model name or path"),
            ]
        )

        if response is None:
            raise typer.BadParameter("Please provide a model name or path.")
        model = response["model"]

    if tasks == []:
        questions = [
            inquirer.Checkbox(
                "tasks",
                message="Select tasks to run",
                choices=[(task.value, task.value) for task in Task],
            ),
        ]

        response = inquirer.prompt(questions)
        if response is None:
            raise typer.BadParameter("Please select at least one task.")
        tasks = response["tasks"]

    slurm_info: Union[None, SlurmInfo] = None
    if slurm:
        if (
            slurm_partition == ""
            or slurm_gres == ""
            or slurm_mem == 0
            or slurm_cpus_per_gpu == 0
        ):
            raise typer.BadParameter(
                "If slurm is enabled, slurm_partition, slurm_gres, slurm_mem, and slurm_cpus_per_gpu must be provided."
            )

        slurm_info = {
            "partition": slurm_partition,
            "gres": slurm_gres,
            "account": slurm_account,
            "mem": slurm_mem,
            "cpus_per_gpu": slurm_cpus_per_gpu,
            "constraint": slurm_constraint,
        }

    services.run(
        model,
        tasks,
        images_directory,
        run_info={
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
        },
        slurm_info=slurm_info,
    )


@app.command()
def setup(
    images_directory: str = "./images",
    definitions: Annotated[List[BenchmarkFramework], typer.Option()] = [],
):
    """
    Setup the environment.

    Args:
        sif_images_directory (str): Directory to store SIF images.
        definitions (List[BenchmarkFrameworks]): List of SIF images to build.
    """

    if definitions == []:
        response = inquirer.prompt(
            [
                inquirer.Checkbox(
                    "sif_images",
                    message="Select SIF images to build",
                    choices=[
                        (benchmark.value, benchmark.value)
                        for benchmark in BenchmarkFramework
                    ],
                ),
            ]
        )
        if response is None:
            raise typer.Exit()
        definitions = response["sif_images"]

    services.setup(images_directory, definitions)


if __name__ == "__main__":
    app()
