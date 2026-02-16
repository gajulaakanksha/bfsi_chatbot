"""Tier 2 -- SLM Engine.

Loads the QLoRA fine-tuned TinyLlama-1.1B-Chat model with 4-bit
quantisation and generates responses for queries that did not match
the curated dataset (Tier 1).

Optionally accepts RAG context to produce grounded answers (Tier 3).
"""
import os
from typing import Optional

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

BASE_MODEL = os.getenv("BASE_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
LORA_PATH = os.getenv("LORA_ADAPTER_PATH", "models/bfsi-lora-adapter")

SYSTEM_PROMPT = (
    "You are a helpful BFSI (Banking, Financial Services, and Insurance) "
    "call center assistant. Provide accurate, concise, and compliant "
    "responses. Never guess financial numbers or rates. If unsure, advise "
    "the customer to contact the branch or helpline."
)

SYSTEM_PROMPT_RAG = (
    "You are a helpful BFSI (Banking, Financial Services, and Insurance) "
    "call center assistant. Use the context provided below to answer the "
    "customer's question. If the request involves drafting content (emails, letters), "
    "structure it as requested while ensuring factual details align with the context. "
    "If the context is irrelevant, state that you cannot answer."
)

# -- Generation defaults --
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
REPETITION_PENALTY = 1.15


class SLMEngine:
    """Load and run the fine-tuned TinyLlama model."""

    def __init__(self, use_lora: bool = True):
        print("[SLMEngine] Loading tokenizer ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[SLMEngine] Loading base model (4-bit) ...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )

        if use_lora and os.path.isdir(LORA_PATH):
            print(f"[SLMEngine] Loading LoRA adapter from {LORA_PATH} ...")
            self.model = PeftModel.from_pretrained(self.model, LORA_PATH)
        else:
            print("[SLMEngine] Running base model (no LoRA adapter found).")

        self.model.eval()
        print("[SLMEngine] Ready.")

    def _build_prompt(self, query: str, rag_context: Optional[str] = None) -> str:
        if rag_context:
            sys_prompt = SYSTEM_PROMPT_RAG
            user_msg = "Context:\n" + rag_context + "\n\nQuestion: " + query
        else:
            sys_prompt = SYSTEM_PROMPT
            user_msg = query
        return (
            "<|system|>\n" + sys_prompt + "</s>\n"
            "<|user|>\n" + user_msg + "</s>\n"
            "<|assistant|>\n"
        )

    @torch.inference_mode()
    def generate(
        self,
        query: str,
        rag_context: Optional[str] = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> str:
        prompt = self._build_prompt(query, rag_context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
        )
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Extract only the assistant response
        marker = "<|assistant|>\n"
        if marker in full:
            response = full.split(marker, 1)[1]
        else:
            response = full[len(prompt):]
        # Clean up end-of-sequence tokens
        for tok in ["</s>", "<|system|>", "<|user|>", "<|assistant|>"]:
            response = response.split(tok)[0]
        return response.strip()
