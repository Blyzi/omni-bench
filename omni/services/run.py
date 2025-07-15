from typing import List, Union
from omni.utils.schemas import RunConfig, SlurmConfig, BenchmarkConfig
from omni.utils.enums import Task, Benchmark
from omni.utils.maps import benchmark_map, task_map
from collections import defaultdict
from rich import print
from uuid import uuid4


def run(
    model: str,
    tasks: List[Task],
    run_config: RunConfig,
    benchmark_config: BenchmarkConfig,
    slurm_config: Union[None, SlurmConfig] = None,
) -> None:
    """
    Run the benchmark.

    Args:
        model (str): The model name or path.
        tasks (List[Task]): The list of tasks to run.
        run_config (RunConfig): Information related to the run.
        slurm_config (Union[None, dict]): SLURM information if applicable.
    """

    run_id = str(uuid4())

    tasks_map = {task: task_map[task] for task in tasks}

    tasks_group = defaultdict(list)
    for key, val in sorted(tasks_map.items()):
        tasks_group[val].append(key)

    frameworks: dict[Benchmark, type] = {}

    for key, val in dict(tasks_group).items():
        frameworks[key] = benchmark_map[key](
            run_id, run_config, benchmark_config, slurm_config
        )

        if frameworks[key].needs_parameters:
            print(
                f"[blue]------------------------ {key} parameters : {','.join(val)} ------------------------[blue]"
            )

            frameworks[key].get_parameters()

    for key, val in dict(tasks_group).items():
        for task in val:
            print(
                f"[blue]------------------------ Running {task} ------------------------[blue]"
            )

            frameworks[key].run(model, task, slurm_config is not None)

    print(
        "[green bold]------------------------ All tasks completed ------------------------[green bold]"
    )
