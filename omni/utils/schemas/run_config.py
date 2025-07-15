from pathlib import Path
from pydantic import Field, BaseModel
from omni.utils.enums import Precision
from omni.utils.schemas.apptainer_bind import ApptainerBind


class RunConfig(BaseModel):
    dtype: Precision = Field(
        description="Data type for model weights and computations, e.g., 'bfloat16', 'float16', or 'float32'",
    )
    tensor_parallel_size: int = Field(
        gt=0,
        description="Size of tensor parallelism, must be greater than 0",
    )
    binds: list[ApptainerBind] = Field(
        default_factory=list,
        description="List of directories to bind in the Apptainer container",
    )
    images_directory: Path = Field(
        description="Directory containing images to include in the Apptainer container",
    )
