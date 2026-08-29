"""
Evaluate OLMo-2-0425-1B on GSM8K using three prompting strategies:
1. Zero-shot question-only (expects answer in \boxed{})
2. Zero-shot R1-Zero reasoning (expects <think>...</think> <answer>...</answer>)
3. Few-shot R1-Zero (3-shot reasoning examples on GSM8K)

Categorizes completions for each prompt style into:
- Category 1: format_reward == 1.0, answer_reward == 1.0 (correct & formatted)
- Category 2: format_reward == 1.0, answer_reward == 0.0 (formatted, incorrect answer)
- Category 3: format_reward == 0.0, answer_reward == 0.0 (unformatted)

Usage Examples:

1. Run full evaluation on Modal:
   uv run modal run scripts/modal_runner.py \
       --command "scripts/prompting_baselines.py"

2. Quick test run with a subset of samples:
   uv run modal run scripts/modal_runner.py \
       --command "scripts/prompting_baselines.py --limit 10"

3. Custom model and output directory:
   uv run modal run scripts/modal_runner.py \
       --command "scripts/prompting_baselines.py --model-id allenai/OLMo-2-0425-1B --output-dir experiments/prompting_baselines"
"""

import typer
from pathlib import Path
from typing import Optional, Any
from collections.abc import Callable
from dataclasses import dataclass
from cs336_alignment import vllm_utils, drgrpo_grader, data_utils


@dataclass(frozen=True)
class PromptConfig:
    """Configuration for a specific prompting strategy and its evaluation setup."""

    name: str
    prompt_path: str
    reward_fn: Callable[..., Any]
    stop_tokens: list[str] | None = None
    response_prefix: str = ""


PROMPT_CONFIGS = [
    PromptConfig(
        name="question_only",
        prompt_path="cs336_alignment/prompts/question_only.prompt",
        reward_fn=drgrpo_grader.question_only_reward_fn,
        stop_tokens=None,  # Uses \boxed{...}.
        response_prefix="",
    ),
    PromptConfig(
        name="r1_zero",
        prompt_path="cs336_alignment/prompts/r1_zero.prompt",
        reward_fn=drgrpo_grader.r1_zero_reward_fn,
        stop_tokens=["</answer>"],
        response_prefix="<think> ",  # The response wouldn't contain this as it is actually being provided in the prompt.
    ),
    PromptConfig(
        name="r1_zero_three_shot",
        prompt_path="cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt",
        reward_fn=drgrpo_grader.r1_zero_reward_fn,
        stop_tokens=["</answer>"],
        response_prefix="<think> ",  # The response wouldn't contain this as it is actually being provided in the prompt.
    ),
]


def main(
    model_id: str = typer.Option(
        "allenai/OLMo-2-0425-1B",
        "--model-id",
        help="Hugging Face model identifier",
    ),
    data_path: str = typer.Option(
        "data/gsm8k/test.jsonl",
        "--data-path",
        help="Path to test data",
    ),
    batch_size: int = typer.Option(
        64,
        "--batch-size",
        help="Batch size for vLLM generation",
    ),
    max_tokens: int = typer.Option(
        512,
        "--max-tokens",
        help="Maximum new tokens to generate",
    ),
    temperature: float = typer.Option(
        1.0,
        "--temperature",
        help="Sampling temperature",
    ),
    num_examples_per_category: int = typer.Option(
        10,
        "--num-examples-per-category",
        help="Number of sample model completions to display for each reward category for qualitative inspection",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Number of test samples to evaluate for quick debugging (default: None, evaluate all)",
    ),
):
    # Load GSM8K test set.
    dataset = data_utils.load_gsm8k(data_path, limit=limit)
    print(f"Loaded {len(dataset)} evaluation samples from {data_path}.\n")

    # Start vLLM server with the model.
    server = vllm_utils.VLLMServer(model_id=model_id)
    server.start()
    try:
        for config in PROMPT_CONFIGS:
            print("=" * 80)
            print(f"Evaluating Prompt Strategy: {config.name}")
            print("=" * 80)

            # Build the prompt from the current strategy's template.
            template = Path(config.prompt_path).read_text()
            prompts = [template.format(question=item.question) for item in dataset]

            sampling_params = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "n": 1,
                "seed": 0,
            }
            if config.stop_tokens is not None:
                sampling_params["stop"] = config.stop_tokens
                sampling_params["include_stop_str_in_output"] = True

            completions = server.generate_completions(
                prompts=prompts,
                sampling_params=sampling_params,
                batch_size=batch_size,
            )

            # Categorize completions.
            cat1 = []  # format == 1.0, answer == 1.0
            cat2 = []  # format == 1.0, answer == 0.0
            cat3 = []  # format == 0.0, answer == 0.0

            for item, comp in zip(dataset, completions):
                reward_dict = config.reward_fn(comp.text, item.ground_truth)
                format_r = reward_dict.get("format_reward", 0.0)
                answer_r = reward_dict.get("answer_reward", 0.0)
                record = {
                    "question": item.question,
                    "ground_truth": item.ground_truth,
                    "response": comp.text,
                    "format_reward": format_r,
                    "answer_reward": answer_r,
                }
                if format_r == 1.0 and answer_r == 1.0:
                    cat1.append(record)
                elif format_r == 1.0 and answer_r == 0.0:
                    cat2.append(record)
                else:
                    cat3.append(record)
            total = len(dataset)
            print(f"\n--- Summary for {config.name} (Total: {total}) ---")
            print(
                f"Category 1 (Format=1, Correct=1): {len(cat1)} ({len(cat1)/total*100:.2f}%)"
            )
            print(
                f"Category 2 (Format=1, Correct=0): {len(cat2)} ({len(cat2)/total*100:.2f}%)"
            )
            print(
                f"Category 3 (Format=0, Correct=0): {len(cat3)} ({len(cat3)/total*100:.2f}%)"
            )
            print(f"Total Accuracy (Answer Reward):  {len(cat1)/total*100:.2f}%\n")

            # Print qualitative examples per category for inspection.
            for cat_name, cat_list in [
                ("Category 1 (Correct & Formatted)", cat1),
                ("Category 2 (Formatted, Incorrect Answer)", cat2),
                ("Category 3 (Unformatted)", cat3),
            ]:
                print(
                    f"\n>>> Examples from {cat_name} (Showing up to {num_examples_per_category}):"
                )
                for i, ex in enumerate(cat_list[:num_examples_per_category]):
                    print(f"\n[Example {i+1}]")
                    print(f"Question:     {ex['question']}")
                    print(f"Ground Truth: {ex['ground_truth']}")
                    print(f"Model Output: {ex['response']}")
                print("-" * 60)
    finally:
        server.stop()


if __name__ == "__main__":
    typer.run(main)
