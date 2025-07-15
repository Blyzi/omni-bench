from global_benchmark.utils.enums import Benchmark
from global_benchmark.benchmarks import (
    LlmEvaluationHarnessRunner,
    BigCodeBenchRunner,
    BigCodeEvaluationHarnessRunner,
)

benchmark_map: dict[Benchmark, type] = {
    Benchmark.LLM_EVALUATION_HARNESS: LlmEvaluationHarnessRunner,
    Benchmark.BIG_CODE_BENCHMARK: BigCodeBenchRunner,
    Benchmark.BIG_CODE_EVALUATION_HARNESS: BigCodeEvaluationHarnessRunner,
}
