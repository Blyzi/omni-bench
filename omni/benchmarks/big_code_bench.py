import json
from pathlib import Path
from typing import List, Literal, Union
import typer
from omni.benchmarks import BenchmarkRunner
import inquirer
import ast
from omni.utils.enums import Benchmark, Task
from rich import print
from omni.utils.schemas import ContainerBind, SlurmConfig, RunConfig, BenchmarkConfig


class BigCodeBenchRunner(BenchmarkRunner):
    needs_parameters = True

    def __init__(
        self,
        run_id: str,
        run_config: RunConfig,
        benchmark_config: BenchmarkConfig,
        slurm_config: SlurmConfig,
    ):
        super().__init__(
            run_id,
            Benchmark.BIG_CODE_BENCHMARK,
            run_config,
            benchmark_config,
            slurm_config,
        )

        print(
            "[yellow bold]WARNING: The benchmark BigCodeBench force the bfloat16 precision, meaning that your GPU must support it or the run will fails[/yellow bold]"
        )

    def get_parameters(self):
        """
        Get the parameters for the benchmark.
        """

        response = inquirer.prompt(
            [
                inquirer.Text(
                    "parameters",
                    message="Enter the benchmark parameters like this: [(temperature, n_samples)]",
                ),
            ]
        )

        if response is None:
            raise typer.BadParameter(
                "No parameters provided. Please provide the parameters in the correct format."
            )

        self.parameters = ast.literal_eval(response["parameters"])

    def run(self, model: str, task: Task, slurm: bool = False):
        """
        Run the benchmark on the BigCode dataset.
        """
        jobs = []

        for temperature, n_samples in self.parameters:
            bench_job = self.exec(
                self.command_wrapper(
                    (
                        "bigcodebench.generate "
                        f"--model {model} "
                        f"--temperature {temperature} "
                        f"--n_samples {n_samples} "
                        f"--split {'complete' if task == Task.BIG_CODE_BENCHMARK_COMPLETE else 'instruct'} "
                        "--subset full "
                        f"--tp {self.run_config.tensor_parallel_size} "
                        "--backend vllm "
                        f"--root /results/temp/{self.run_id}/{task}_{temperature}_{n_samples} "
                    ),
                    container=True,
                    slurm=slurm,
                    slurm_partition="gpu",
                ),
                slurm=slurm,
            )

            copy_job = self.exec(
                self.command_wrapper(
                    f"cp -r ./results/temp/{self.run_id}/{task}_{temperature}_{n_samples} ./results/temp/{self.run_id}/{task}_hard_{temperature}_{n_samples}",
                    container=False,
                    slurm=slurm,
                    slurm_partition="cpu",
                    slurm_dependency=[bench_job],
                ),
                slurm=slurm,
            )

            full_eval_job = self.exec(
                self.command_wrapper(
                    (
                        "bigcodebench.evaluate "
                        f"--model {model} "
                        f"--temperature {temperature} "
                        f"--n_samples {n_samples} "
                        f"--samples /results/temp/{self.run_id}/{task}_{temperature}_{n_samples}/{model.replace('/', '--')}--main--bigcodebench-{'complete' if task == Task.BIG_CODE_BENCHMARK_COMPLETE else 'instruct'}--vllm-{temperature}-{n_samples}-sanitized_calibrated.jsonl "
                        "--execution local "
                        f"--split {'complete' if task == Task.BIG_CODE_BENCHMARK_COMPLETE else 'instruct'} "
                        "--subset full "
                        "--backend vllm "
                    ),
                    container=True,
                    benchmark_eval=True,
                    slurm=slurm,
                    slurm_partition="cpu",
                    slurm_dependency=[copy_job],
                ),
                slurm=slurm,
            )

            hard_eval_job = self.exec(
                self.command_wrapper(
                    (
                        "bigcodebench.evaluate "
                        f"--model {model} "
                        f"--temperature {temperature} "
                        f"--n_samples {n_samples} "
                        f"--samples /results/temp/{self.run_id}/{task}_hard_{temperature}_{n_samples}/{model.replace('/', '--')}--main--bigcodebench-{'complete' if task == Task.BIG_CODE_BENCHMARK_COMPLETE else 'instruct'}--vllm-{temperature}-{n_samples}-sanitized_calibrated.jsonl "
                        "--execution local "
                        f"--split {'complete' if task == Task.BIG_CODE_BENCHMARK_COMPLETE else 'instruct'} "
                        "--subset hard "
                        "--backend vllm "
                    ),
                    container=True,
                    benchmark_eval=True,
                    slurm=slurm,
                    slurm_partition="cpu",
                    slurm_dependency=[copy_job],
                ),
                slurm=slurm,
            )

            jobs.extend([full_eval_job, hard_eval_job])

        self.exec(
            self.command_wrapper(
                f"uv run omni save {self.run_id} {task} {model}",
                container=False,
                slurm=slurm,
                slurm_partition="cpu",
                slurm_dependency=jobs,
            ),
            slurm=slurm,
        )

    def command_wrapper(
        self,
        command: str,
        container: bool,
        slurm: bool,
        slurm_partition: Union[Literal["cpu"], Literal["gpu"]],
        slurm_dependency: List[str] = [],
        benchmark_eval: bool = False,
    ):
        """
        Execute a command in the container.
        """

        if container:
            command = self.get_container_command(
                command,
                self.framework.value + ("_eval" if benchmark_eval else "_gen"),
                [ContainerBind(source=Path("results"), target=Path("/results"))],
            )

        if slurm:
            command = self.get_slurm_command(command, slurm_partition, slurm_dependency)

        return command

    def save(
        self,
        model: str,
        task: Task,
    ) -> None:
        """
        Save the results of the benchmark.
        """

        run_path = Path("results/temp") / self.run_id

        for folder in run_path.glob(f"{task.value}*"):
            for result_file in folder.glob("*_pass_at_k.json"):
                with open(result_file, "r") as f:
                    data = json.load(f)

                subset, temperature, n_samples = folder.name.split("_")[-3:]

                self.store(
                    model,
                    task,
                    {
                        "pass@1": data["pass@1"],
                        "subset": subset,
                        "temperature": temperature,
                        "n_samples": n_samples,
                    },
                )
