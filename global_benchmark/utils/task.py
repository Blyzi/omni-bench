from enum import Enum


class Task(str, Enum):
    """
    Enum class for benchmark names.
    """

    # LlmEvaluationHarness
    MBPP = "mbpp"
    MBPP_PLUS = "mbpp_plus"

    HUMANEVAL = "humaneval"
    HUMANEVAL_PLUS = "humaneval_plus"
    HUMANEVAL_INSTRUCT = "humaneval_instruct"

    MMLU = "mmlu"

    ARC_EASY = "arc_easy"
    ARC_EASY_CHAT = "arc_easy_chat"
    ARC_CHALLENGE = "arc_challenge"
    ARC_CHALLENGE_CHAT = "arc_challenge_chat"

    HELLASWAG = "hellaswag"

    IFEVAL = "ifeval"

    XNLI = "xnli"

    BELEBELE = "belebele"

    # BigCodeBenchmark
    BIG_CODE_BENCHMARK_COMPLETE = "big_code_bench_complete"
    BIG_CODE_BENCHMARK_INSTRUCT = "big_code_bench_instruct"

    # BigCodeEvaluationHarness
    MULTIPLE = "multiple"
    APPS = "apps"
