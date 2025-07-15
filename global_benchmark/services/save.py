from typing import Union
from global_benchmark.utils.enums import Task
from global_benchmark.utils.maps import benchmark_map, task_map
from global_benchmark.utils.schemas import RunConfig, SlurmConfig


def save(
    run_id: str,
    model: str,
    task: Task,
    run_config: RunConfig,
    slurm_config: Union[None, SlurmConfig] = None,
):
    """
    Save the results of the benchmark.
    """

    framework = benchmark_map[task_map[task]](run_id, run_config, slurm_config)

    framework.save(model, task)
