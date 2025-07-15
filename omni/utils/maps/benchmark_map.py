from omni.utils.enums import Benchmark
from omni.benchmarks import (
    LlmEvaluationHarnessRunner,
    BigCodeBenchRunner,
    BigCodeEvaluationHarnessRunner,
    RulerRunner,
)

benchmark_map: dict[Benchmark, type] = {
    Benchmark.LLM_EVALUATION_HARNESS: LlmEvaluationHarnessRunner,
    Benchmark.BIG_CODE_BENCHMARK: BigCodeBenchRunner,
    Benchmark.BIG_CODE_EVALUATION_HARNESS: BigCodeEvaluationHarnessRunner,
    Benchmark.RULER: RulerRunner,
}
