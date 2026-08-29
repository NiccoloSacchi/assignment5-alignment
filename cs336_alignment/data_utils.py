import json
from typing import Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    question: str
    raw_answer: str
    ground_truth: str


def load_gsm8k(data_path: str, limit: Optional[int] = None) -> list[Sample]:
    """Loads GSM8K test samples and extracts the clean numerical ground truth."""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            # In GSM8K, the target answer is after '####'.
            ground_truth = item["answer"].split("####")[-1].strip()
            samples.append(
                Sample(
                    question=item["question"],
                    ground_truth=ground_truth,
                    raw_answer=item["answer"],
                ),
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples
