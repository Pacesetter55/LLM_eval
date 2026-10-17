# eval_retriever.py
import asyncio
import json
import os
import sys
from pathlib import Path

# deepeval's overall gather timeout doesn't scale with case count; with throttled
# concurrency 100 cases outlast it, so give the gather step a generous buffer.
os.environ.setdefault("DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE", "1800")

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, ErrorConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retriever import get_retriever, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from evals.qwen_call import get_chat_response

# qwen_call posts with verify=False; silence the per-request InsecureRequestWarning spam.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL_NAME = "qwen (self-hosted)"
JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge. Follow the instructions exactly. "
    "When asked for JSON, reply with only valid JSON: no markdown fences, no extra text."
)
THRESHOLD = 0.7
MAX_CONCURRENT = 2   # each in-flight case fires several judge calls; raise if the Qwen server has headroom
JSON_REPAIR_ATTEMPTS = 2  # follow-up "fix your JSON" calls before giving up on a verdict


class QwenJudge(DeepEvalBaseLLM):
    """deepeval judge backed by our own deployed Qwen endpoint (see qwen_call.py)."""

    def __init__(self):
        super().__init__(model=JUDGE_MODEL_NAME)

    def load_model(self):
        return self

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        text = get_chat_response(prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
        if schema is None:
            return text
        # deepeval passes a pydantic schema for structured verdicts; parse it ourselves.
        # Temperature is 0, so a plain retry repeats the same broken JSON (usually unescaped
        # quotes in a "reason"); instead feed the parse error back and ask Qwen to repair it.
        for attempt in range(JSON_REPAIR_ATTEMPTS + 1):
            start, end = text.find("{"), text.rfind("}")
            try:
                if start == -1 or end <= start:
                    raise ValueError("no JSON object found in the response")
                return schema.model_validate_json(text[start:end + 1])
            except ValueError as e:
                if attempt == JSON_REPAIR_ATTEMPTS:
                    break
                text = get_chat_response(
                    "Your previous reply was not valid JSON.\n"
                    f"Parse error: {e}\n\n"
                    f"Previous reply:\n{text}\n\n"
                    "Return the same content as valid JSON only. Escape any double quotes "
                    "inside string values as \\\" and do not add markdown fences or extra text.",
                    system_prompt=JUDGE_SYSTEM_PROMPT,
                )
        # Still broken: hand the raw text to deepeval's own JSON trimming as a last resort.
        return text

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        # get_chat_response is blocking (requests); run it in a thread so concurrency still works.
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return JUDGE_MODEL_NAME


JUDGE_MODEL = QwenJudge()


def load_goldens(path: str) -> list[dict]:
    """Load the JSONL golden dataset (one JSON object per line)."""
    goldens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                goldens.append(json.loads(line))
    return goldens


def summarize_by_metric(result) -> dict:
    """Return pass-rate per metric name from a deepeval EvaluationResult."""
    totals: dict[str, list[bool]] = {}
    for test_result in result.test_results:
        for metric_data in test_result.metrics_data:
            name = metric_data.name
            totals.setdefault(name, []).append(bool(metric_data.success))  # errored metrics may carry None
    return {
        name: {"passed": sum(vals), "total": len(vals), "pass_rate": sum(vals) / len(vals)}
        for name, vals in totals.items()
    }


def print_summary(label: str, summary: dict) -> None:
    print(f"\n=== {label} eval summary ===")
    for metric, stats in summary.items():
        print(f"  {metric}: {stats['passed']}/{stats['total']} passed ({stats['pass_rate']:.1%})")


def run(retriever):
    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    test_cases = []
    for g in goldens:
        retrieved = retriever.invoke(g["question"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["question"],
                expected_output=g["reference_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(max_concurrent=MAX_CONCURRENT),
        # one unparseable judge verdict shouldn't abort the whole run; that metric is
        # recorded as errored (failed) on its case and the rest carry on.
        error_config=ErrorConfig(ignore_errors=True),
        hyperparameters={
            "retriever": "base",
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": 4,
            "judge_model": JUDGE_MODEL_NAME,
            "golden_set": GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the retriever, then run."""
    return run(get_retriever())


if __name__ == "__main__":
    print_summary("retriever", run_local())