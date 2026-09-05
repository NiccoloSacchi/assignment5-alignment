import torch
from typing import Callable, Literal
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from jaxtyping import Float
from einops import rearrange, einsum


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


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    assert (
        raw_rewards.shape[0] % group_size == 0
    ), "raw_rewards shape must be a multiple of group_size"

    # Reshape into (num_groups, group_size) so all groups are processed in
    # parallel.
    grouped = rearrange(raw_rewards, "(g s) -> g s", s=group_size)

    # 1. Compute baseline (shape: (num_groups, 1) or scalar).
    if baseline == "mean":
        b = grouped.mean(dim=-1, keepdim=True)
    elif baseline == "none":
        b = 0.0
    else:
        raise NotImplementedError(f"baseline '{baseline}' not supported")

    # 2. Compute normalizer (shape: (num_groups, 1) or scalar).
    if advantage_normalizer == "std":
        norm = grouped.std(dim=-1, keepdim=True) + advantage_eps
    elif advantage_normalizer == "mean":
        norm = grouped.mean(dim=-1, keepdim=True) + advantage_eps
    elif advantage_normalizer == "none":
        norm = 1.0
    else:
        raise NotImplementedError(
            f"advantage_normalizer '{advantage_normalizer}' not supported"
        )

    # 3. Compute advantages and flatten back to 1D.
    advantages = rearrange((grouped - b) / norm, "g s -> (g s)")
    metadata = {
        "mean_advantage": advantages.mean().item(),
        "std_advantage": advantages.std().item() if len(advantages) > 1 else 0.0,
    }
    return advantages, metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: (
        Float[torch.Tensor, "batch_size"] | Float[torch.Tensor, "batch_size 1"]
    ),
    policy_log_probs: Float[torch.Tensor, "batch_size sequence_length"],
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Float[torch.Tensor, "batch_size sequence_length"] | None = None,
    cliprange: float | None = None,
    response_mask: Float[torch.Tensor, "batch_size sequence_length"] | None = None,
) -> tuple[Float[torch.Tensor, "batch_size sequence_length"], dict[str, torch.Tensor]]:
    """
    uv run pytest -k test_compute_policy_gradient_loss_on_policy
    """

    if importance_reweighting_method != "none":
        raise NotImplementedError(
            f"importance_reweighting_method {importance_reweighting_method} not supported"
        )

    per_token_loss = -einsum(
        raw_rewards_or_advantages.view(-1),
        policy_log_probs,
        "batch_size, batch_size sequence_length -> batch_size sequence_length",
    )

    metadata = {}
    return per_token_loss, metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: Float[torch.Tensor, "batch_size sequence_length"],
    mask: Float[torch.Tensor, "batch_size sequence_length"],
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> Float[torch.Tensor, "1"]:
    """
    uv run pytest -k test_aggregate_loss_across_microbatch_sequence
    """
    per_token_loss_masked = per_token_policy_gradient_loss * mask

    loss = None
    if loss_normalization == "sequence":
        # Compute the normalization per-sequence, depending on the amount of
        # non-masked tokens.
        per_sequence_loss = per_token_loss_masked.sum(dim=-1) / mask.sum(dim=-1)
        loss = per_sequence_loss.mean()
    elif loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError(
                "normalization_constant must be provided for constant normalization"
            )
        loss = per_token_loss_masked.sum() / normalization_constant
    else:
        raise NotImplementedError(
            f"normalization_constant {normalization_constant} not supported"
        )

    return loss
