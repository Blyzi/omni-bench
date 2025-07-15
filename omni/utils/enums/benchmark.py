from enum import Enum


class Benchmark(str, Enum):
    """
    Enum class for Benchmarks.
    """

    LLM_EVALUATION_HARNESS = "llm_evaluation_harness"
    BIG_CODE_BENCHMARK = "big_code_bench"
    BIG_CODE_EVALUATION_HARNESS = "big_code_evaluation_harness"
    RULER = "ruler"
