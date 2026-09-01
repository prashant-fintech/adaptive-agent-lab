"""Answer the same prompt with the base model and with the polite adapter,
side by side.

Ordering constraint: attaching a PEFT adapter mutates the loaded model, so
the base answer is generated FIRST, then the adapter is attached and the
polite answer generated from the same weights + LoRA delta.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_agent import config
from adaptive_agent.finetune.train_qlora import apply_template

MAX_NEW_TOKENS = 200


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return tokenizer, model


def attach_polite_adapter(model):
    from peft import PeftModel

    if not config.ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"{config.ADAPTER_DIR} missing - train it first (08_train_adapter)"
        )
    return PeftModel.from_pretrained(model, str(config.ADAPTER_DIR))


def generate(tokenizer, model, prompt: str) -> str:
    text = apply_template(
        tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def compare(prompt: str) -> None:
    tokenizer, model = load_base()

    print(f'prompt: "{prompt}"\n')
    print("--- base model ---")
    print(generate(tokenizer, model, prompt), "\n")

    model = attach_polite_adapter(model)
    print("--- polite adapter ---")
    print(generate(tokenizer, model, prompt))
