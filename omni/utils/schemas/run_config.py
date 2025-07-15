from pathlib import Path
from typing import Optional
from pydantic import Field, BaseModel
from omni.utils.enums import Precision, ContainerSystem
from omni.utils.schemas.container_bind import ContainerBind


class RunConfig(BaseModel):
    container_system: ContainerSystem = Field(
        default=ContainerSystem.APPTAINER,
        description="Container system to use, e.g., 'apptainer' or 'singularity'",
    )
    dtype: Precision = Field(
        description="Data type for model weights and computations, e.g., 'bfloat16', 'float16', or 'float32'",
    )
    tensor_parallel_size: int = Field(
        gt=0,
        description="Size of tensor parallelism, must be greater than 0",
    )
    binds: list[ContainerBind] = Field(
        default_factory=list,
        description="List of directories to bind in the container",
    )
    images_directory: Path = Field(
        description="Directory containing images to include in the container",
    )
