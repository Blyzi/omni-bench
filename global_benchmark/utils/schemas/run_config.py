import json
from pathlib import Path
from typing import Final
from pydantic import Field, BaseModel, ValidationError, model_validator
import typer
from global_benchmark.utils.enums import Precision
from global_benchmark.utils.schemas.apptainer_bind import ApptainerBind


class RunConfig(BaseModel):
    dtype: Precision = Field(
        default=Precision.BF16,
        description="Data type for model weights and computations, e.g., 'bfloat16', 'float16', or 'float32'",
    )
    tensor_parallel_size: int = Field(
        default=1,
        gt=0,
        description="Size of tensor parallelism, must be greater than 0",
    )
    binds: list[ApptainerBind] = Field(
        default_factory=list,
        description="List of directories to bind in the Apptainer container",
    )
    images_directory: Path = Field(
        default=Path("images"),
        description="Directory containing images to include in the Apptainer container",
    )

    REQUIRED_FIELDS: Final = [
        "dtype",
        "tensor_parallel_size",
        "binds",
        "images_directory",
    ]

    @classmethod
    def load(cls, config: dict) -> "RunConfig":
        # Validate that required fields are explicitly present
        missing = [field for field in cls.REQUIRED_FIELDS if field not in config]
        if missing:
            raise typer.BadParameter(
                f"Missing required fields in config file: {', '.join(missing)}"
            )

        return cls(**config)
