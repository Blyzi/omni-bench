from typing import TypedDict


class RunInfo(TypedDict):
    dtype: str
    tensor_parallel_size: int
