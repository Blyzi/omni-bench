# Global Benchmarks

> I think nobody wants their time spent on running benchmarks. Personally I don't, that why I created global benchmark during my LLM benchmarking internship at INRIA. - Theo Lasnier

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
uv run global-benchmark config
```

The command will install all the dependencies and create a default configuration file `config.json`. You can modify this file to change the default settings for the project.

## Usage

### Configuration

The configuration file `config.json` is used to define the default settings for the project. You can modify this file to change the default settings for the project. The configuration file is structured with a run config, which is used to define the default settings for the benchmarks, and a slurm config, which is used to define the default settings for running benchmarks on a SLURM cluster.

#### Run Config
- `dtype`: The default model precision to use. Must be one of `float16`, `bfloat16`, or `float32`.
- `tp`: The default tensor parallelism to use. Must be an integer greater than or equal to 1.
- `images_directory`: The directory containing the images to be used for the benchmarks. The default is `./images`.
- `binds`: The list of binds to use for the container. This is used to bind the host directories to the container directories. You have to bind the models folder so that the benchmarks can access the models.

#### SLURM Config
- `cpu_partition`: The SLURM partition to use for CPU tasks.
- `cpus_per_task`: The number of CPUs per task to use for CPU tasks.
- `gpu_partition`: The SLURM partition to use for GPU tasks.
- `gpu_gres`: The SLURM generic resources to use for GPU tasks. This is used to specify the number of GPUs to use.
- `mem`: The amount of memory to use for the tasks. This is used to specify the amount of memory to allocate for the tasks.
- `cpus_per_gpu`: The number of CPUs per GPU to use for GPU tasks.
- `concurrency`: (Optional) The number of concurrent tasks to run. This is used to specify the number of tasks to run in parallel.
- `account`: (Optional) The SLURM account to use for the tasks. This is used to specify the account to charge the resources to.
- `gpu_constraint`: (Optional) The GPU constraint to use for the tasks. This is used to specify the type of GPU to use.
- `exclude`: (Optional) The list of nodes to exclude from the tasks. This is used to specify the nodes that should not be used for the tasks.

### Create environment

For reproducibility and security purpose, each framework is run in a apptainer container. To build the container, run the following command:

```bash
uv run global-benchmark setup
```

Parameters:
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
- `--slurm`(Optional): Use SLURM for job scheduling. This is useful for running benchmarks on a cluster. The default is `false`.

### Results
The results of the benchmarks are stored in the `results` directory. The results are stored in JSON format, making it easy to parse and analyze them.