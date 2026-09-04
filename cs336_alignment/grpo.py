import torch
from typing import Callable
from transformers import PreTrainedTokenizerBase, PreTrainedModel


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """
    uv run pytest -k test_tokenize_prompt_and_output
    """

    # Encode prompts and outputs and compute mask.
    prompt_and_outputs = []
    masks = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_encoded = tokenizer.encode(
            prompt, skip_special_tokens=True, add_special_tokens=False
        )
        output_encoded = tokenizer.encode(
            output, skip_special_tokens=True, add_special_tokens=False
        )
        prompt_and_outputs.append(prompt_encoded + output_encoded)
        masks.append([0] * (len(prompt_encoded) - 1) + [1] * len(output_encoded))

    # Pad them to same len.
    max_len = len(max(prompt_and_outputs, key=len))
    prompt_and_outputs_padded = []
    mask_padded = []
    for i in range(len(prompt_and_outputs)):
        pad_len = max_len - len(prompt_and_outputs[i])
        prompt_and_outputs_padded.append(
            prompt_and_outputs[i] + [tokenizer.pad_token_id] * pad_len
        )
        mask_padded.append(masks[i] + [0] * pad_len)

    prompt_and_outputs_tensor = torch.tensor(
        prompt_and_outputs_padded, dtype=torch.long
    )
    return {
        "input_ids": prompt_and_outputs_tensor[:, :-1],
        "labels": prompt_and_outputs_tensor[:, 1:],
        "response_mask": torch.tensor(mask_padded, dtype=torch.int),
    }


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    uv run pytest -k test_get_response_log_probs
    """
    out = model(input_ids.to(model.device))

    # Shape: [batch_size, seq_len, vocab_size].
    log_probs = torch.log_softmax(out.logits, dim=-1)

    # Pick out the log-prob corresponding to each label token.
    selected_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1))
    selected_log_probs = selected_log_probs.squeeze(-1)  # [batch_size, seq_len]

    if return_token_entropy:
        return {
            "log_probs": selected_log_probs,
            # Shape: [batch_size, seq_len].
            "token_entropy": -(log_probs.exp() * log_probs).sum(dim=-1),
        }
    return {
        "log_probs": selected_log_probs,
    }


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    uv run pytest -k compute_rollout_rewards
    """
    rewards = []
    format_rewards = []
    answer_rewards = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(resp, gt)
        rewards.append(scores["reward"])
        format_rewards.append(scores["format_reward"])
        answer_rewards.append(scores["answer_reward"])

    raw_rewards = torch.tensor(rewards, dtype=torch.float32)
    format_t = torch.tensor(format_rewards, dtype=torch.float32)
    answer_t = torch.tensor(answer_rewards, dtype=torch.float32)
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_format_reward": format_t.mean().item(),
        "mean_answer_reward": answer_t.mean().item(),
    }
    return raw_rewards, metadata
