from enum import Enum


class Precision(str, Enum):
    BF16 = "bfloat16"
    FP16 = "float16"
    FP32 = "float32"
