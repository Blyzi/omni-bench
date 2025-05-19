from enum import Enum


class BenchmarkFramework(str, Enum):
    """
    Enum class for BenchmarkFrameworks.
    """

    LLM_EVALUATION_HARNESS = "llm_evaluation_harness"
    BIG_CODE_BENCHMARK = "big_code_bench"
    BIG_CODE_EVALUATION_HARNESS = "big_code_evaluation_harness"
