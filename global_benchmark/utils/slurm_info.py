from typing import TypedDict


class SlurmInfo(TypedDict):
    partition: str
    gres: str
    account: str
    mem: str
    cpus_per_gpu: int
    constraint: str
