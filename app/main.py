"""Streamlit entry point: `streamlit run app/main.py`"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import app.*` works under streamlit

import streamlit as st

from app.core import load_jsonl

ADAPTER_DIR = "adapters/commentary-lora"
DATA_PATH = "data/commentary.jsonl"

st.set_page_config(page_title="Sports Commentary Generator")
st.title("Sports Commentary Generator")
st.write("Game developers use this tool to adapt a language model with LoRA, generating dynamic and "
         "context-aware commentary snippets for simulated sports events like goal alerts or score changes.")


@st.cache_resource(show_spinner="Loading model...")
def get_model(adapter_dir: str, merge: bool):
    from app.lora import load_for_inference
    return load_for_inference(adapter_dir, merge=merge)


with st.sidebar:
    st.header("Settings")
    adapter_dir = st.text_input("Adapter folder", ADAPTER_DIR)
    merge = st.checkbox("Merge adapter for inference", value=True,
                        help="Folds the LoRA weights into GPT-2 so generation runs as one plain model.")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.8, 0.05, help="0 = deterministic")
    max_tokens = st.slider("Max new tokens", 10, 80, 40)

has_adapter = Path(adapter_dir).exists()
if not has_adapter:
    st.warning(f"No adapter found at `{adapter_dir}`. Run `python scripts/train.py` first. "
               "Until then you are talking to plain GPT-2.")

examples = load_jsonl(DATA_PATH) if Path(DATA_PATH).exists() else []
if examples and st.button("Pick a random example event"):
    st.session_state["event"] = random.choice(examples)["event"]

event = st.text_area("Event", key="event", height=80,
                     placeholder="Football | 78' | Goal | Rivera (Red Lions) | Red Lions 2-1 Blue Hawks")

if st.button("Generate commentary", type="primary", disabled=not event.strip()):
    from app.lora import generate
    model, tok = get_model(adapter_dir if has_adapter else "", merge)
    with st.spinner("Commentating..."):
        text = generate(model, tok, event, max_tokens, temperature)
    st.subheader("Commentary")
    st.success(text or "(the model produced nothing, try again)")
    st.caption(("Merged LoRA model" if merge else "LoRA adapter (unmerged)") if has_adapter else "Base GPT-2")
