from global_benchmark.utils.schemas.run_config import Precision


def get_short_precision(precision: Precision) -> str:
    """
    Get the precision string for the given precision type. Must be able to go from short to long and vice versa.

    Args:
            precision (str): The precision type (e.g., 'float32', 'bfloat16').

    Returns:
            str: The corresponding precision string.
    """
    precision_map = {
        Precision.FP32: "fp32",
        Precision.BF16: "bf16",
        Precision.FP16: "fp16",
    }

    if precision not in precision_map:
        raise ValueError(
            f"Invalid precision type: {precision}. Must be one of {list(precision_map.keys())}."
        )

    return precision_map[precision]
