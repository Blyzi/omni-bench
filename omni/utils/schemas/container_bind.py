from pathlib import Path
from pydantic import BaseModel, Field


class ContainerBind(BaseModel):
    source: Path = Field(
        description="Source directory to bind in the container, e.g., './scratch/model'",
    )
    target: Path = Field(
        description="Target directory in the container, e.g., '/models'",
    )
