import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase

try:
    from deepeval.test_case import SingleTurnParams as eval_params
except ImportError:
    from deepeval.test_case import LLMTestCaseParams as eval_params


load_dotenv(".env.local")
load_dotenv(".env")

data_file = Path("data.csv")
results_file = Path("dresults.csv")

openai_api_key = os.getenv("OPENAI_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openai_api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in .env.local or environment variables.")
if not openrouter_api_key:
    raise RuntimeError("Missing OPENROUTER_API_KEY in .env.local or environment variables.")

openai_client = OpenAI(api_key=openai_api_key)
openrouter_client = OpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1")

models = [
    {"name": "gpt-5.4-nano", "provider": "openai", "model": "gpt-5.4-nano"},
    {"name": "gemini-3.1-flash-lite", "provider": "openrouter", "model": "google/gemini-3.1-flash-lite"},
    {"name": "claude-haiku-4.5", "provider": "openrouter", "model": "anthropic/claude-haiku-4.5"},
    {"name": "deepseek-v4-flash", "provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
]

result_fields = [
    "model_name", "provider", "model", "id", "category", "question", "context", "expected",
    "actual_output", "answer_relevancy_score", "answer_relevancy_passed",
    "answer_relevancy_reason", "factual_correctness_score", "factual_correctness_passed",
    "factual_correctness_reason", "error",
]


def call_model(provider: str, model: str, question: str) -> str:
    client = openai_client if provider == "openai" else openrouter_client if provider == "openrouter" else None
    if client is None:
        raise ValueError(f"Unknown provider: {provider}")

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Atbildi latviešu valodā. Sniedz īsu, precīzu un faktoloģiski korektu atbildi.",
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content.strip()


def create_metrics():
    answer_relevancy = AnswerRelevancyMetric(threshold=0.7)
    factual_correctness = GEval(
        name="Factual Correctness",
        criteria=(
            "Evaluate whether the actual output correctly answers the input question and is "
            "consistent with the expected output. Penalize hallucinated or unsupported claims."
        ),
        evaluation_params=[
            eval_params.INPUT,
            eval_params.ACTUAL_OUTPUT,
            eval_params.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
    )
    return answer_relevancy, factual_correctness


def get_metric_result(metric, threshold: float = 0.7):
    score = getattr(metric, "score", None)
    reason = getattr(metric, "reason", "")
    try:
        passed = metric.is_successful()
    except Exception:
        passed = score is not None and score >= threshold
    return score, reason, passed


def load_test_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def create_result(model_config: dict, row: dict, actual_output: str = "", error: str = "") -> dict:
    return {
        "model_name": model_config["name"],
        "provider": model_config["provider"],
        "model": model_config["model"],
        "id": row["id"],
        "category": row["category"],
        "question": row["question"],
        "context": row.get("context", ""),
        "expected": row["expected"],
        "actual_output": actual_output,
        "answer_relevancy_score": "",
        "answer_relevancy_passed": False,
        "answer_relevancy_reason": "",
        "factual_correctness_score": "",
        "factual_correctness_passed": False,
        "factual_correctness_reason": "",
        "error": error,
    }


def evaluate_row(model_config: dict, row: dict) -> dict:
    actual_output = call_model(model_config["provider"], model_config["model"], row["question"])
    test_case = LLMTestCase(
        input=row["question"],
        actual_output=actual_output,
        expected_output=row["expected"],
    )

    answer_relevancy, factual_correctness = create_metrics()
    answer_relevancy.measure(test_case)
    factual_correctness.measure(test_case)

    ar_score, ar_reason, ar_passed = get_metric_result(answer_relevancy)
    fc_score, fc_reason, fc_passed = get_metric_result(factual_correctness)

    result = create_result(model_config, row, actual_output)
    result.update({
        "answer_relevancy_score": ar_score,
        "answer_relevancy_passed": ar_passed,
        "answer_relevancy_reason": ar_reason,
        "factual_correctness_score": fc_score,
        "factual_correctness_passed": fc_passed,
        "factual_correctness_reason": fc_reason,
    })
    return result


def main():
    rows = load_test_data(data_file)
    results = []

    for model_config in models:
        print(f"\n=== Testing model: {model_config['name']} ===")

        for row in rows:
            print(f"Running test {row['id']}: {row['question']}")
            try:
                results.append(evaluate_row(model_config, row))
            except Exception as error:
                results.append(create_result(model_config, row, error=str(error)))
                print(f"Error in test {row['id']}: {error}")

    if not results:
        print("No results were produced.")
        return

    with results_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
