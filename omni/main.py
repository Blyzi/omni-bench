from pathlib import Path
from typing import List
import typer
import inquirer
from typing_extensions import Annotated
from omni.utils.enums import Task, Benchmark
from omni import services
import os


os.environ["HF_ALLOW_CODE_EVAL"] = "1"

app = typer.Typer()


@app.command()
def run(
    model: Annotated[str, typer.Option("--model", "-m")] = "",
    tasks: Annotated[List[Task], typer.Option("--tasks", "-t")] = [],
    slurm: bool = False,
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

    services.run(
        model,
        tasks,
        run_config=services.get_run_config(),
        slurm_config=services.get_slurm_config() if slurm else None,
    )


@app.command()
def setup(
    images_directory: Annotated[Path, typer.Option("--images-directory", "-i")] = Path(
        "images"
    ),
    definitions: Annotated[List[Benchmark], typer.Option()] = [],
):
    """
    Setup the environment.

    Args:
        images_directory (str): Directory to store images.
        definitions (List[Benchmarks]): List of images to build.
    """

    if definitions == []:
        response = inquirer.prompt(
            [
                inquirer.Checkbox(
                    "images",
                    message="Select images to build",
                    choices=[
                        (benchmark.value, benchmark.value) for benchmark in Benchmark
                    ],
                ),
            ]
        )
        if response is None:
            raise typer.Exit()
        definitions = response["images"]

    services.setup(images_directory, definitions)


@app.command()
def config():
    """
    Generate a base configuration file.
    """

    services.gen_config()


@app.command()
def list():
    """
    List all available tasks.
    """

    typer.echo("Available tasks:")
    for task in Task:
        typer.echo(task.value)


@app.command(hidden=True)
def save(
    run_id: str,
    task: Task,
    model: str,
):
    """
    Save the results of the benchmark.
    """

    services.save(
        run_id,
        model,
        task,
        run_config=services.get_run_config(),
        slurm_config=services.get_slurm_config(),
    )


if __name__ == "__main__":
    app()
