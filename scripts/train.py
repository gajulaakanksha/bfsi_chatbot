"""QLoRA fine-tuning script for TinyLlama-1.1B-Chat on BFSI Alpaca dataset.

Fine-tunes using 4-bit quantisation so the model fits within 8 GB VRAM
(RTX 3070 Laptop GPU). Produces LoRA adapter weights saved to the path
configured in .env (LORA_ADAPTER_PATH).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

load_dotenv()

# -- Configuration -----------------------------------------------------
BASE_MODEL = os.getenv("BASE_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
LORA_OUTPUT = os.getenv("LORA_ADAPTER_PATH", "models/bfsi-lora-adapter")
DATASET_PATH = os.path.join("data", "alpaca_bfsi_dataset.json")

MAX_SEQ_LEN   = 512
BATCH_SIZE    = 2
GRAD_ACCUM    = 4     # effective batch = 8
EPOCHS        = 3
LEARNING_RATE = 2e-4
WARMUP_RATIO  = 0.05
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05

SYSTEM_PROMPT = (
    "You are a helpful BFSI (Banking, Financial Services, and Insurance) "
    "call center assistant. Provide accurate, concise, and compliant "
    "responses. Never guess financial numbers or rates. If unsure, advise "
    "the customer to contact the branch or helpline."
)


# -- Format function ---------------------------------------------------
def format_alpaca(sample):
    """Convert an Alpaca-format sample into a TinyLlama chat prompt."""
    instruction = sample["instruction"]
    inp = sample.get("input", "")
    output = sample["output"]
    user_msg = instruction + ("\n\nContext: " + inp if inp else "")
    return (
        "<|system|>\n" + SYSTEM_PROMPT + "</s>\n"
        "<|user|>\n" + user_msg + "</s>\n"
        "<|assistant|>\n" + output + "</s>"
    )


# -- Main --------------------------------------------------------------
def main():
    print("=" * 60)
    print("QLoRA Fine-Tuning: TinyLlama-1.1B-Chat  ->  BFSI Assistant")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1/5] Loading dataset...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ds = Dataset.from_list(raw)
    print(f"  Loaded {len(ds)} samples from {DATASET_PATH}")

    # 2. Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Load model with 4-bit quantisation
    print("\n[3/5] Loading base model with 4-bit quantisation...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 4. Configure LoRA
    print("\n[4/5] Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Train
    print("\n[5/5] Starting training...")
    training_args = SFTConfig(
        output_dir="models/training_checkpoints",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        fp16=False,
        bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        max_grad_norm=0.3,
        group_by_length=True,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        formatting_func=format_alpaca,
        args=training_args,
    )

    trainer.train()

    # 6. Save adapter
    print("\n Saving LoRA adapter...")
    os.makedirs(LORA_OUTPUT, exist_ok=True)
    model.save_pretrained(LORA_OUTPUT)
    tokenizer.save_pretrained(LORA_OUTPUT)
    print(f"  Adapter saved to: {LORA_OUTPUT}")
    print("\n  Fine-tuning complete!")


if __name__ == "__main__":
    main()
