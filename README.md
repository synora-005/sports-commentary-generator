# Sports Commentary Generator

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/y49dqs38ha/badge.svg)](https://msrit.ask.thili.ai/b/y49dqs38ha)

Adapts GPT-2 with **LoRA** to turn a simulated sports event into a short piece of live commentary, with a Streamlit UI
that runs the **merged** model.

## How it works
1. `app/core.py` builds a dataset of (event, commentary) pairs (`data/commentary.jsonl`, 300 rows, 5 sports) and formats
   each as `Event: ...\nCommentary: ...<|endoftext|>`.
2. `app/lora.py` freezes GPT-2, injects LoRA (r=8, alpha=16, on `c_attn`), and trains with a plain PyTorch loop. Only the
   commentary tokens contribute to the loss. The adapter (about 1 MB) is saved to `adapters/commentary-lora`.
3. `app/main.py` is the Streamlit UI (event box, temperature and max-token sliders, merge toggle).
4. `scripts/evaluate.py` compares base GPT-2 with the fine-tuned model on held-out events and writes `EVALUATION.md`.
5. `merge_adapter()` calls `merge_and_unload()` so inference uses one plain model. A test proves merged and unmerged
   outputs match.

## Run it
```bash
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/train.py --epochs 5     # downloads GPT-2 once; a few minutes on a CPU
python scripts/evaluate.py             # base vs LoRA, writes EVALUATION.md
streamlit run app/main.py
pytest                                 # fast, offline (uses a tiny random GPT-2)
```
No API keys or `.env` file are needed: hosted models cannot be LoRA-trained, so everything runs locally.

## Example input
`Football | 78' | Goal | Rivera (Red Lions) | Red Lions 2-1 Blue Hawks`

## Milestones
- [x] 1. Dataset (`app/core.py`, `data/commentary.jsonl`)
- [x] 2. LoRA injection + training (`app/lora.py`, `scripts/train.py`)
- [x] 3. Streamlit interface (`app/main.py`)
- [x] 4. Evaluation (`scripts/evaluate.py`, `EVALUATION.md`)
- [x] 5. Adapter merging (`merge_adapter`, `load_for_inference`)

## Results
Held-out events (30), greedy decoding, base GPT-2 vs the merged LoRA model (details in `EVALUATION.md`):

| Model | Mean relevance (player/team mentioned) | Well-formed |
|---|---|---|
| Base GPT-2 (baseline) | 15% | 20% |
| LoRA (merged) | 88% | 100% |

LoRA training adds roughly 0.3M trainable parameters to GPT-2's 124M and lifts relevance from 15% to 88%.

## Limitations
- The dataset is template-generated, so the model learns those phrasings and can sound repetitive.
- GPT-2 is small: it can swap team names or invent scores for events unlike the training data.
- Relevance is a keyword check (player and team mentioned), not a measure of how engaging the text is.
- Only 5 sports and 20 event types are covered; other sports give weaker output.
- Held-out events come from the same templates as the training data, so real-world performance will be lower.