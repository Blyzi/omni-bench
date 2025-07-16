from typing import Union
from omni.utils.enums import Task
from omni.utils.maps import benchmark_map, task_map
from omni.utils.schemas import RunConfig, SlurmConfig, BenchmarkConfig
from filelock import FileLock


def save(
    run_id: str,
    model: str,
    task: Task,
    run_config: RunConfig,
    benchmark_config: BenchmarkConfig,
    slurm_config: Union[None, SlurmConfig] = None,
):
    """
    Save the results of the benchmark.
    """

    framework = benchmark_map[task_map[task]](
        run_id, run_config, benchmark_config, slurm_config
    )

    with FileLock(f"results/temp/{run_id}/.lock"):
        framework.save(model, task)
