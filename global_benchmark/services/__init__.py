from global_benchmark.services.run import run
from global_benchmark.services.setup import setup
from global_benchmark.services.config import (
    get_run_config,
    get_slurm_config,
    gen_config,
)
from global_benchmark.services.save import save

__all__ = [
    "run",
    "setup",
    "get_run_config",
    "get_slurm_config",
    "gen_config",
    "save",
]
