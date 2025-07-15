from pathlib import Path
from pydantic import BaseModel, Field


class ApptainerBind(BaseModel):
    source: Path = Field(
        description="Source directory to bind in the Apptainer container, e.g., './scratch/model'",
    )
    target: Path = Field(
        description="Target directory in the Apptainer container, e.g., '/models'",
    )
