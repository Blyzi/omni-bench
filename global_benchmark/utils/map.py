from global_benchmark.benchmarks import (
    BigCodeBenchRunner,
    BigCodeEvaluationHarnessRunner,
    LlmEvaluationHarnessRunner,
)
from global_benchmark.utils import BenchmarkFramework, Task


benchmark_framework_map: dict[BenchmarkFramework, type] = {
    BenchmarkFramework.LLM_EVALUATION_HARNESS: LlmEvaluationHarnessRunner,
    BenchmarkFramework.BIG_CODE_BENCHMARK: BigCodeBenchRunner,
    BenchmarkFramework.BIG_CODE_EVALUATION_HARNESS: BigCodeEvaluationHarnessRunner,
}

task_map: dict[Task, BenchmarkFramework] = {
    # LlmEvaluationHarness
    Task.HUMANEVAL: BenchmarkFramework.LLM_EVALUATION_HARNESS,
    Task.HUMANEVAL_PLUS: BenchmarkFramework.LLM_EVALUATION_HARNESS,
    Task.HUMANEVAL_INSTRUCT: BenchmarkFramework.LLM_EVALUATION_HARNESS,
    Task.MBPP: BenchmarkFramework.LLM_EVALUATION_HARNESS,
    Task.MBPP_PLUS: BenchmarkFramework.LLM_EVALUATION_HARNESS,
    Task.MMLU: BenchmarkFramework.LLM_EVALUATION_HARNESS,
    # BigCodeBenchmark
    Task.BIG_CODE_BENCHMARK_COMPLETE: BenchmarkFramework.BIG_CODE_BENCHMARK,
    Task.BIG_CODE_BENCHMARK_INSTRUCT: BenchmarkFramework.BIG_CODE_BENCHMARK,
    # BigCodeEvaluationHarness
    Task.MULTIPLE: BenchmarkFramework.BIG_CODE_EVALUATION_HARNESS,
    Task.APPS: BenchmarkFramework.BIG_CODE_EVALUATION_HARNESS,
}
