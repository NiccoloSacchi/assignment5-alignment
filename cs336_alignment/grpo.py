import torch
from transformers import PreTrainedTokenizerBase


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
