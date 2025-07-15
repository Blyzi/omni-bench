from pydantic import BaseModel, Field
from omni.utils.enums import Benchmark


class LlmEvaluationHarnessConfig(BaseModel):
    add_bos_token: bool = Field(
        default=False,
        description="Whether to add a beginning-of-sequence token to the input text.",
    )


class RulerConfig(BaseModel):
    """
    Configuration for the Ruler benchmark.
    """

    thread_count: int = Field(
        default=4,
        description="Number of threads to use for the benchmark.",
    )


class BenchmarkConfig(BaseModel):
    """
    Configuration for the benchmark.
    """

    llm_evaluation_harness: LlmEvaluationHarnessConfig = Field(
        default_factory=LlmEvaluationHarnessConfig,
        alias=Benchmark.LLM_EVALUATION_HARNESS.value,
        serialization_alias=Benchmark.LLM_EVALUATION_HARNESS.value,
    )

    ruler: RulerConfig = Field(
        default_factory=RulerConfig,
        alias=Benchmark.RULER.value,
        serialization_alias=Benchmark.RULER.value,
    )
