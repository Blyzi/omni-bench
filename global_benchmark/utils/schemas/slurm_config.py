from typing import Final, Optional, Annotated
from pydantic import BaseModel, Field
import typer


class SlurmConfig(BaseModel):
    cpu_partition: str = Field(
        default="cpu", description="Partition for CPU jobs, e.g., 'cpu'"
    )
    cpus_per_task: int = Field(
        default=16,
        gt=0,
        description="Number of CPUs allocated for each job, must be greater than 0",
    )
    gpu_partition: str = Field(
        default="gpu", description="Partition for GPU jobs, e.g., 'gpu'"
    )
    gpu_gres: str = Field(
        default="gpu:1", description="GPU resources, e.g., 'gpu:1' for one GPU"
    )
    mem: str = Field(
        default="64G", description="Memory allocated for each job, e.g., '64G'"
    )
    cpus_per_gpu: int = Field(
        default=16, gt=0, description="Number of CPUs per GPU, must be greater than 0"
    )
    account: Optional[
        Annotated[str, Field(description="Slurm account name for job submission")]
    ] = None
    gpu_constraint: Optional[
        Annotated[str, Field(description="Slurm constraint for job scheduling")]
    ] = None
    exclude: Optional[
        Annotated[str, Field(description="Nodes to exclude from job scheduling")]
    ] = None

    REQUIRED_FIELDS: Final = [
        "cpu_partition",
        "cpus_per_task",
        "gpu_partition",
        "gpu_gres",
        "mem",
        "cpus_per_gpu",
    ]

    @classmethod
    def load(cls, config: dict) -> "SlurmConfig":
        # Validate that required fields are explicitly present
        missing = [field for field in cls.REQUIRED_FIELDS if field not in config]
        if missing:
            raise typer.BadParameter(
                f"Missing required fields in config file: {', '.join(missing)}"
            )

        return cls(**config)
