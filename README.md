# Global Benchmarks

This project aims to provide an global interface and environment for running benchmarks on various models. It is designed to be extensible, allowing anyone to modify the benchmarks or add new ones. The goal is to create a unified framework for benchmarking different models, making it easier to compare their performance and capabilities.

## Frameworks
The project currently supports the following frameworks/benchmarks:
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main)
- [bigcodebench](https://github.com/bigcode-project/bigcodebench/tree/main)

If you would like to add support for a new framework, please follow the instructions in the [CONTRIBUTING.md](CONTRIBUTING.md) file. The process is designed to be straightforward, allowing you to easily integrate your framework into the project.

## Installation
This project use [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management. To install the project depencies, run the following command:

```bash
uv sync
```

## Usage

### Create environment

For reproducibility and security purpose, each framework is run in a apptainer container. To create the container, run the following command:

```bash
uv run global-benchmark setup
```

Parameters:
- `--images-directory`(Optional): The directory containing the images to be used for the benchmarks. Be careful, some images are quite large (up to 20GB). The default is `./images`.
- `--definitions`(Optional): The list of definitions files to build. You can use this parameter or the integrated CLI if you prefer.

**Warning**: This command will take a while to run, as it needs to download the images and build the containers. The images are quite large, so make sure you have enough disk space available.

### Run benchmarks
To run the benchmarks, use the following command:

```bash
uv run global-benchmark run
```
Parameters:
- `--model`(Optional): The model to be used for the benchmarks. This can be a local model or a remote model. You can use the CLI to define it if you prefer.
- `--tasks`(Optional): The list of tasks to run. You can use the CLI to define them if you prefer.
- `--dtype`(Optional): Model precision when possible (some frameworks may not support this). The default is `auto`.
- `--tp`(Optional): The number of tensor parallelism. The default is `1`.
- `--images-directory`(Optional): The directory containing the images to be used for the benchmarks from the setup step. The default is `./images`.
- `--slurm`(Optional): Use SLURM for job scheduling. This is useful for running benchmarks on a cluster. The default is `false`.
- `--slurm-partition`(Optional): The SLURM partition to use. Mandatory if `--slurm` is set to `true`.
- `--slurm-gres`(Optional): The SLURM generic resources to use. Mandatory if `--slurm` is set to `true`.
- `--slurm-cpus-per-gpu`(Optional): The number of CPUs per GPU to use. Mandatory if `--slurm` is set to `true`.
- `--slurm-mem`(Optional): The amount of memory to use. Mandatory if `--slurm` is set to `true`.
- `--slurm-account`(Optional): The SLURM account to use. 

### Results
The results of the benchmarks are stored in the `results` directory. The results are stored in JSON format, making it easy to parse and analyze them.

## Adding a new framework or adding a new benchmark

### Adding a new benchmark
If one of the supported frameworks supports a benchmark that is not currently supported, you can add it by following these steps:
1. Go to the `task.py` file and add your tasks to the enum `Task`.
2. Go to the `map.py` file and map your new tasks to the appropriate benchmark frameworks.

All apptainer images are sandboxed, meaning you can directly modify the content of the container. This is useful if you want to do modify or create new benchmarks.

### Adding a new framework
If you want to add a new framework, you can do so by following these steps:
1. Create a definition file for your framework in the `definitions` directory. This file should contain the necessary information for creating the container.
2. Create a new file in the `benchmarks` directory for your framework. This file should contain the necessary code for running the benchmarks for your framework.
    1. Implement the `exec` function that should run a command in the container.
    2. Implement the `run` function that should run the benchmarks for your framework.
    3. If necessary, implement the `get_parameters` function that will prompt the user for the parameters to use for the benchmarks.

3. Go to the `benchmark_framework.py` file and add your new framework to the `BenchmarkFramework` enum.