"""LoRA training, generation and adapter merging (Milestones 2 and 5). Needs torch, transformers, peft."""
import random
from pathlib import Path

from app.core import clean_commentary, format_example, format_prompt

BASE_MODEL = "gpt2"
LORA_TARGETS = ["c_attn"]  # GPT-2's fused attention projection (query, key, value)


def load_base(name: str = BASE_MODEL):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token  # GPT-2 has no pad token
    return AutoModelForCausalLM.from_pretrained(name), tok


# ---------------------------------------------------------------- Milestone 2
def add_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    """Inject LoRA adapters: base weights are frozen, only the small A/B matrices train."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS,
                     fan_in_fan_out=True,  # GPT-2 uses Conv1D, whose weights are stored transposed
                     bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def count_params(model) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad), total


def encode_rows(tok, rows: list[dict], max_len: int = 96) -> list[dict]:
    """Tokenize rows. Labels are -100 (ignored) on the prompt, so the model learns the commentary only."""
    out = []
    for r in rows:
        prompt_ids = tok(format_prompt(r["event"]))["input_ids"]
        ids = tok(format_example(r["event"], r["commentary"]))["input_ids"][:max_len]
        labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
        out.append({"input_ids": ids, "labels": labels[:len(ids)]})
    return out


def make_batches(encoded: list[dict], batch_size: int, pad_id: int) -> list[dict]:
    """Pad to the longest item in each batch and return tensors."""
    import torch

    batches = []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i:i + batch_size]
        width = max(len(c["input_ids"]) for c in chunk)
        ids = [c["input_ids"] + [pad_id] * (width - len(c["input_ids"])) for c in chunk]
        lab = [c["labels"] + [-100] * (width - len(c["labels"])) for c in chunk]
        mask = [[1] * len(c["input_ids"]) + [0] * (width - len(c["input_ids"])) for c in chunk]
        batches.append({"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                        "attention_mask": torch.tensor(mask)})
    return batches


def train_loop(model, batches: list[dict], epochs: int = 5, lr: float = 2e-4, seed: int = 0) -> list[float]:
    """Plain PyTorch loop. Returns the mean loss of each epoch."""
    import torch

    rng = random.Random(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    history = []
    for epoch in range(epochs):
        rng.shuffle(batches)
        total = 0.0
        for batch in batches:
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            total += loss.item()
        history.append(total / len(batches))
        print(f"epoch {epoch + 1}/{epochs}  loss {history[-1]:.4f}")
    return history


def train(rows: list[dict], adapter_dir: str = "adapters/commentary-lora", epochs: int = 5,
          lr: float = 2e-4, batch_size: int = 8, base: str = BASE_MODEL) -> list[float]:
    model, tok = load_base(base)
    model = add_lora(model)
    trainable, total = count_params(model)
    print(f"training {trainable:,} of {total:,} parameters ({100 * trainable / total:.2f}%)")
    batches = make_batches(encode_rows(tok, rows), batch_size, tok.pad_token_id)
    history = train_loop(model, batches, epochs, lr)
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)  # saves only the adapter (about 1 MB)
    return history


# ---------------------------------------------------------------- Milestone 5
def merge_adapter(base_model, adapter_dir: str):
    """Fold the LoRA weights into the base weights: one plain model, no adapter overhead at inference."""
    from peft import PeftModel

    return PeftModel.from_pretrained(base_model, adapter_dir).merge_and_unload()


def load_for_inference(adapter_dir: str | None, merge: bool = True, base: str = BASE_MODEL):
    """Return (model, tokenizer). No adapter_dir -> plain base model (used as the comparison baseline)."""
    model, tok = load_base(base)
    if adapter_dir and Path(adapter_dir).exists():
        if merge:
            model = merge_adapter(model, adapter_dir)
        else:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tok


def generate(model, tok, event: str, max_new_tokens: int = 40, temperature: float = 0.8, top_p: float = 0.9) -> str:
    import torch

    model.eval()
    enc = tok(format_prompt(event), return_tensors="pt")
    kwargs = {"do_sample": True, "temperature": temperature, "top_p": top_p} if temperature > 0 else {"do_sample": False}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id,
                             eos_token_id=tok.eos_token_id, **kwargs)
    return clean_commentary(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
