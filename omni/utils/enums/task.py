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
    MULTIPLE_CLJCPP = "multiple-cljcpp"
    MULTIPLE_CS = "multiple-cs"
    MULTIPLE_D = "multiple-d"
    MULTIPLE_DART = "multiple-dart"
    MULTIPLE_ELIXIR = "multiple-elixir"
    MULTIPLE_GO = "multiple-go"
    MULTIPLE_HS = "multiple-hs"
    MULTIPLE_JAVA = "multiple-java"
    MULTIPLE_JL = "multiple-jl"
    MULTIPLE_JS = "multiple-js"
    MULTIPLE_LUA = "multiple-lua"
    MULTIPLE_MLPL = "multiple-mlpl"
    MULTIPLE_PHP = "multiple-php"
    MULTIPLE_PY = "multiple-py"
    MULTIPLE_R = "multiple-r"
    MULTIPLE_RB = "multiple-rb"
    MULTIPLE_RKT = "multiple-rkt"
    MULTIPLE_RS = "multiple-rs"
    MULTIPLE_SCALA = "multiple-scala"
    MULTIPLE_SH = "multiple-sh"
    MULTIPLE_SWIFT = "multiple-swift"
    MULTIPLE_TS = "multiple-ts"

    APPS_INTRODUCTORY = "apps-introductory"
    APPS_INTERVIEW = "apps-interview"
    APPS_COMPETITION = "apps-competition"

    HUMANEVALFIXDOCS_PYTHON = "humanevalfixdocs-python"
    HUMANEVALFIXTESTS_PYTHON = "humanevalfixtests-python"
    HUMANEVALEXPLAINDESCRIBE_PYTHON = "humanevalexplaindescribe-python"
    HUMANEVALEXPLAINSYNTHESIZE_PYTHON = "humanevalexplainsynthesize-python"
    HUMANEVALSYNTHESIZE_PYTHON = "humanevalsynthesize-python"
    HUMANEVALFIXDOCS_CPP = "humanevalfixdocs-cpp"
    HUMANEVALFIXTESTS_CPP = "humanevalfixtests-cpp"
    HUMANEVALEXPLAINDESCRIBE_CPP = "humanevalexplaindescribe-cpp"
    HUMANEVALEXPLAINSYNTHESIZE_CPP = "humanevalexplainsynthesize-cpp"
    HUMANEVALSYNTHESIZE_CPP = "humanevalsynthesize-cpp"
    HUMANEVALFIXDOCS_JS = "humanevalfixdocs-js"
    HUMANEVALFIXTESTS_JS = "humanevalfixtests-js"
    HUMANEVALEXPLAINDESCRIBE_JS = "humanevalexplaindescribe-js"
    HUMANEVALEXPLAINSYNTHESIZE_JS = "humanevalexplainsynthesize-js"
    HUMANEVALSYNTHESIZE_JS = "humanevalsynthesize-js"
    HUMANEVALFIXDOCS_JAVA = "humanevalfixdocs-java"
    HUMANEVALFIXTESTS_JAVA = "humanevalfixtests-java"
    HUMANEVALEXPLAINDESCRIBE_JAVA = "humanevalexplaindescribe-java"
    HUMANEVALEXPLAINSYNTHESIZE_JAVA = "humanevalexplainsynthesize-java"
    HUMANEVALSYNTHESIZE_JAVA = "humanevalsynthesize-java"
    HUMANEVALFIXDOCS_GO = "humanevalfixdocs-go"
    HUMANEVALFIXTESTS_GO = "humanevalfixtests-go"
    HUMANEVALEXPLAINDESCRIBE_GO = "humanevalexplaindescribe-go"
    HUMANEVALEXPLAINSYNTHESIZE_GO = "humanevalexplainsynthesize-go"
    HUMANEVALSYNTHESIZE_GO = "humanevalsynthesize-go"
    HUMANEVALFIXDOCS_RUST = "humanevalfixdocs-rust"
    HUMANEVALFIXTESTS_RUST = "humanevalfixtests-rust"
    HUMANEVALEXPLAINDESCRIBE_RUST = "humanevalexplaindescribe-rust"
    HUMANEVALEXPLAINSYNTHESIZE_RUST = "humanevalexplainsynthesize-rust"
    HUMANEVALSYNTHESIZE_RUST = "humanevalsynthesize-rust"

    # Ruler
    RULER_SYNTHETIC = "ruler_synthetic"
