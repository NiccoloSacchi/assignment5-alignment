import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    is_cuda = "cuda" in device
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if is_cuda else "sdpa",  # or "eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer
