from typing import Optional, Annotated
from pydantic import BaseModel, ConfigDict, Field


class SlurmJobConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


class SlurmConfig(BaseModel):
    cpu_partition: SlurmJobConfig
    gpu_partition: SlurmJobConfig

    prescript: Optional[str] = Field(
        default=None,
        description="Optional script to run before the main command in the container",
    )
