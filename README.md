# Sports Commentary Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/y49dqs38ha/badge.svg)](https://ask.thili.ai/b/y49dqs38ha)

This project provides game developers with a tool to generate dynamic and context-aware sports commentary. By adapting a pre-trained language model using the LoRA fine-tuning technique, it creates realistic commentary snippets for simulated sports events like goal alerts or score changes. The tool proves the effectiveness of LoRA in specializing LLMs for specific domain tasks.

## What you are building

**Goal.** You will have built a tool that adapts a language model with LoRA to generate dynamic and context-aware commentary snippets for simulated sports events, demonstrating the LoRA fine-tuning technique.

**Done when.** The project is done when the LoRA fine-tuned model can generate relevant and engaging sports commentary snippets for simulated events, and the Streamlit interface successfully demonstrates this generation after adapter merging.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Create a small dataset of simulated sports events and commentary pairs.**  
      Hint: Focus on data preparation and formatting for fine-tuning.
- [ ] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: Apply the `LoRA adapter injection` and `training loop` techniques from the drill.
- [ ] **3. Build a Streamlit interface to input event descriptions and generate commentary.**  
      Hint: Integrate your `LoRA-adapted model` into a Streamlit user interface.
- [ ] **4. Test the model's ability to produce relevant and engaging commentary.**  
      Hint: Evaluate the `generated text` for quality, relevance, and engagement.
- [ ] **5. Implement adapter merging for inference.**  
      Hint: Use the `adapter merging` technique from the drill for efficient inference.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- (none)
