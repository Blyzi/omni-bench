from enum import Enum


class ContainerSystem(str, Enum):
    """
    Enum class for Container Systems.
    """

    APPTAINER = "apptainer"
    SINGULARITY = "singularity"
