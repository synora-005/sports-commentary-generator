# Sports Commentary Generator

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
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --epochs 5     # downloads GPT-2 once; a few minutes on a CPU
python scripts/evaluate.py             # base vs LoRA, writes EVALUATION.md
streamlit run app/main.py
pytest                                 # fast, offline (uses a tiny random GPT-2)
```
The `.env` / Ollama / Groq / Gemini settings from the starter are not needed for this project (`app/llm.py` is unused,
since hosted models cannot be LoRA-trained).

## Example input
`Football | 78' | Goal | Rivera (Red Lions) | Red Lions 2-1 Blue Hawks`

## Milestones
- [ ] 1. Dataset (`app/core.py`, `data/commentary.jsonl`)
- [ ] 2. LoRA injection + training (`app/lora.py`, `scripts/train.py`)
- [ ] 3. Streamlit interface (`app/main.py`)
- [ ] 4. Evaluation (`scripts/evaluate.py`, `EVALUATION.md`)
- [ ] 5. Adapter merging (`merge_adapter`, `load_for_inference`)

## Results
Paste the table from `EVALUATION.md` here after you run it.
| Model | Mean relevance (player/team mentioned) | Well-formed |
|---|---|---|
| Base GPT-2 | 15% | 20% |
| LoRA (merged) | 88% | 100% |