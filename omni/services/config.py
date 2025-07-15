import os
from pathlib import Path
import yaml
from pydantic import ValidationError
import typer
from omni.utils.enums import Precision
from omni.utils.schemas import SlurmConfig, RunConfig, SlurmJobConfig, BenchmarkConfig


def get_run_config() -> RunConfig:
    """
    Get the run config from the config.yml file or use the default values.
    """
    if os.path.exists("config.yml"):
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)

            if not isinstance(config, dict) or "run" not in config:
                raise typer.BadParameter("config.yml does not contain 'run' section.")
    else:
        raise typer.BadParameter(
            "config.yml file not found. Please generate it using `omni config`."
        )

    try:
        config = RunConfig(**config["run"])
    except ValidationError as e:
        raise typer.BadParameter(f"\n{e}")

    return config


def get_slurm_config() -> SlurmConfig:
    """
    Get the Slurm config from the config.json file or use the default values.
    """
    if os.path.exists("config.yml"):
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)

            if not isinstance(config, dict) or "slurm" not in config:
                raise typer.BadParameter("config.yml does not contain 'slurm' section.")
    else:
        raise typer.BadParameter(
            "config.yml file not found. Please generate it using `omni config`."
        )

    try:
        config = SlurmConfig(**config["slurm"])
    except ValidationError as e:
        raise typer.BadParameter(f"\n{e}")

    return config


def gen_config():
    """
    Generate a config.yml file with default values.
    """

    if os.path.exists("config.yml"):
        print("config.yml already exists. Please delete it to generate a new one.")
        return

    with open("config.yml", "w") as f:
        yaml.safe_dump(
            {
                "run": RunConfig(
                    dtype=Precision.BF16,
                    tensor_parallel_size=1,
                    binds=[],
                    images_directory=Path("images"),
                ).model_dump(mode="json"),
                "slurm": SlurmConfig(
                    cpu_partition=SlurmJobConfig(),
                    gpu_partition=SlurmJobConfig(),
                ).model_dump(mode="json"),
                "benchmarks": BenchmarkConfig().model_dump(mode="json", by_alias=True),
            },
            f,
            indent=4,
        )
