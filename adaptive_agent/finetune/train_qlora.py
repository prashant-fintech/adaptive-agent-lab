"""QLoRA fine-tune of Qwen3-0.6B into the "super polite" adapter.

The pipeline is a chain of named stages, each with an inspectable return
value:

    load_pairs -> to_chat_example -> tokenize_example -> train

Only the LoRA matrices train (about 1-2% of parameters); the base weights
stay frozen. With a CUDA GPU the base loads in 4-bit (QLoRA). On CPU or
without bitsandbytes, pass --no-4bit: slower, same resulting adapter.
"""

import json

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from adaptive_agent import config

MAX_LENGTH = 512


def load_pairs() -> list[dict]:
    path = config.POLITE_PAIRS_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run 07_build_polite_dataset first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def apply_template(tokenizer, messages: list[dict], add_generation_prompt: bool) -> str:
    # Qwen3 accepts enable_thinking; older templates do not - fall back cleanly.
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )


def tokenize_example(tokenizer, pair: dict) -> dict:
    """Token ids for prompt+answer, with the prompt tokens masked out of the
    loss (label -100) so the model is only trained on how it ANSWERS."""
    user = [{"role": "user", "content": pair["question"]}]
    full = user + [{"role": "assistant", "content": pair["answer"]}]

    prompt_text = apply_template(tokenizer, user, add_generation_prompt=True)
    full_text = apply_template(tokenizer, full, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:MAX_LENGTH]

    labels = [-100] * min(len(prompt_ids), len(full_ids))
    labels += full_ids[len(labels) :]
    return {"input_ids": full_ids, "labels": labels}


def collate_batch(tokenizer, batch: list[dict]) -> dict:
    longest = max(len(example["input_ids"]) for example in batch)
    input_ids, labels, attention = [], [], []
    for example in batch:
        pad = longest - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [tokenizer.pad_token_id] * pad)
        labels.append(example["labels"] + [-100] * pad)
        attention.append([1] * len(example["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention),
    }


def load_base_model(use_4bit: bool):
    if use_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_ID, quantization_config=quant_config, device_map="auto"
        )
    return AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_ID)


def add_lora(model, use_4bit: bool):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(use_4bit: bool = True, epochs: int = 2, learning_rate: float = 2.0e-4) -> None:
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)
    pairs = load_pairs()
    dataset = [tokenize_example(tokenizer, pair) for pair in pairs]
    print(f"training on {len(dataset)} examples from {config.POLITE_PAIRS_PATH}")

    model = add_lora(load_base_model(use_4bit), use_4bit)

    arguments = TrainingArguments(
        output_dir=str(config.ARTIFACTS_DIR / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=use_4bit,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=lambda batch: collate_batch(tokenizer, batch),
    )
    trainer.train()

    model.save_pretrained(config.ADAPTER_DIR)
    tokenizer.save_pretrained(config.ADAPTER_DIR)
    print(f"adapter saved -> {config.ADAPTER_DIR}")
