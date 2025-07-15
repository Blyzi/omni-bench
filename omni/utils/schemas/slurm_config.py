from typing import Optional, Annotated
from pydantic import BaseModel, ConfigDict, Field


class SlurmJobConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


class SlurmConfig(BaseModel):
    cpu_partition: SlurmJobConfig
    gpu_partition: SlurmJobConfig

    account: Optional[
        Annotated[str, Field(description="Slurm account name for job submission")]
    ] = None
