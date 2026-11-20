"""Build the dataset (if missing), train the LoRA adapter, optionally save a merged model.

    python scripts/train.py --epochs 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import generate_dataset, load_jsonl, save_jsonl, split_dataset
from app.lora import load_base, merge_adapter, train

DATA, ADAPTER, MERGED = "data/commentary.jsonl", "adapters/commentary-lora", "models/merged"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n", type=int, default=300, help="dataset size if it must be generated")
    ap.add_argument("--save-merged", action="store_true", help="also write the merged model to models/merged")
    args = ap.parse_args()

    if not Path(DATA).exists():
        save_jsonl(generate_dataset(args.n), DATA)
        print(f"wrote {DATA}")
    train_rows, _held_out = split_dataset(load_jsonl(DATA))  # held-out rows are never trained on
    train(train_rows, ADAPTER, epochs=args.epochs, lr=args.lr)
    print(f"adapter saved to {ADAPTER}")
    if args.save_merged:
        model, tok = load_base()
        merge_adapter(model, ADAPTER).save_pretrained(MERGED)
        tok.save_pretrained(MERGED)
        print(f"merged model saved to {MERGED}")
