"""Milestone 4: compare plain GPT-2 with the fine-tuned (merged) model on held-out events.

    python scripts/evaluate.py          # writes EVALUATION.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import is_well_formed, load_jsonl, relevance_score, split_dataset
from app.lora import generate, load_for_inference

DATA, ADAPTER, N = "data/commentary.jsonl", "adapters/commentary-lora", 30

if __name__ == "__main__":
    _train, held_out = split_dataset(load_jsonl(DATA))
    held_out = held_out[:N]
    results = {}
    for name, adapter in (("Base GPT-2", None), ("LoRA (merged)", ADAPTER)):
        model, tok = load_for_inference(adapter, merge=True)
        results[name] = [generate(model, tok, r["event"], temperature=0) for r in held_out]  # greedy: repeatable
    lines = [f"# Evaluation on {len(held_out)} held-out events\n",
             "| Model | Mean relevance (player/team mentioned) | Well-formed |", "|---|---|---|"]
    for name, outs in results.items():
        rel = sum(relevance_score(r["keywords"], o) for r, o in zip(held_out, outs)) / len(outs)
        ok = sum(is_well_formed(o) for o in outs) / len(outs)
        lines.append(f"| {name} | {rel:.0%} | {ok:.0%} |")
    lines.append("\n## Examples\n")
    for i, r in enumerate(held_out[:5]):
        lines += [f"**Event:** {r['event']}", ""]
        lines += [f"- {name}: {outs[i] or '(empty)'}" for name, outs in results.items()] + [""]
    Path("EVALUATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
