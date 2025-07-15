from omni.services.run import run
from omni.services.setup import setup
from omni.services.config import (
    get_run_config,
    get_slurm_config,
    gen_config,
)
from omni.services.save import save

__all__ = [
    "run",
    "setup",
    "get_run_config",
    "get_slurm_config",
    "gen_config",
    "save",
]
