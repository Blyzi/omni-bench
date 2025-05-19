from typing import List, Union
from global_benchmark.utils import (
    RunInfo,
    SlurmInfo,
    Task,
    BenchmarkFramework,
    task_map,
    benchmark_framework_map,
)
from collections import defaultdict
from rich import print
from uuid import uuid4


def run(
    model: str,
    tasks: List[Task],
    images_directory: str,
    run_info: RunInfo,
    slurm_info: Union[None, SlurmInfo] = None,
) -> None:
    """
    Run the benchmark.

    Args:
        model (str): The model name or path.
        tasks (List[Task]): The list of tasks to run.
        run_info (RunInfo): Information related to the run.
        slurm_info (Union[None, dict]): SLURM information if applicable.
    """

    run_id = str(uuid4())

    tasks_map = {task: task_map[task] for task in tasks}

    tasks_group = defaultdict(list)
    for key, val in sorted(tasks_map.items()):
        tasks_group[val].append(key)

    frameworks: dict[BenchmarkFramework, type] = {}

    for key, val in dict(tasks_group).items():
        frameworks[key] = benchmark_framework_map[key](
            run_id, run_info, slurm_info, images_directory
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

            frameworks[key].run(model, task, slurm_info is not None)

    print(
        "[green bold]------------------------ All tasks completed ------------------------[green bold]"
    )
