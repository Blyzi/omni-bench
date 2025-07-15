import os
import json
from pydantic import ValidationError
import typer
from omni.utils.schemas import SlurmConfig, RunConfig


def get_run_config() -> RunConfig:
    """
    Get the run config from the config.json file or use the default values.
    """
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)

            if not isinstance(config, dict) or "run" not in config:
                raise typer.BadParameter("config.json does not contain 'run' section.")
    else:
        raise typer.BadParameter(
            "config.json file not found. Please generate it using `omni config`."
        )

    try:
        config = RunConfig.load(config["run"])
    except ValidationError as e:
        raise typer.BadParameter(f"\n{e}")

    return config


def get_slurm_config() -> SlurmConfig:
    """
    Get the Slurm config from the config.json file or use the default values.
    """
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)

            if not isinstance(config, dict) or "slurm" not in config:
                raise typer.BadParameter(
                    "config.json does not contain 'slurm' section."
                )
    else:
        raise typer.BadParameter(
            "config.json file not found. Please generate it using `omni config`."
        )

    try:
        config = SlurmConfig.load(config["slurm"])
    except ValidationError as e:
        raise typer.BadParameter(f"\n{e}")

    return config


def gen_config():
    """
    Generate a config.json file with default values.
    """

    if os.path.exists("config.json"):
        print("config.json already exists. Please delete it to generate a new one.")

    with open("config.json", "w") as f:
        json.dump(
            {
                "run": RunConfig().model_dump(mode="json"),
                "slurm": SlurmConfig().model_dump(mode="json"),
            },
            f,
            indent=4,
        )
