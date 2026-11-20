"""One test per milestone. LoRA tests use a tiny random GPT-2 built locally: no downloads, runs in CI."""
import pytest

from app.core import (EOS, clean_commentary, format_example, format_prompt, generate_dataset,
                      is_well_formed, load_jsonl, relevance_score, save_jsonl, split_dataset)


# Milestone 1: dataset
def test_milestone_1_dataset_is_unique_deterministic_and_well_formatted(tmp_path):
    rows = generate_dataset(200, seed=1)
    assert len(rows) == 200
    assert len({(r["event"], r["commentary"]) for r in rows}) == 200
    assert rows == generate_dataset(200, seed=1)
    assert all(r["keywords"][0] in r["commentary"] for r in rows)  # commentary mentions the actor
    text = format_example("Football | 5' | Goal | Rivera (Red Lions) | Red Lions 1-0 Blue Hawks", "GOAL!")
    assert text.startswith("Event: Football") and text.endswith(f"Commentary: GOAL!{EOS}")
    assert format_prompt("x") == "Event: x\nCommentary:"
    save_jsonl(rows, str(tmp_path / "d.jsonl"))
    assert load_jsonl(str(tmp_path / "d.jsonl")) == rows
    train, held = split_dataset(rows)
    assert held and len(train) + len(held) == len(rows)
    assert not ({id(r) for r in train} & {id(r) for r in held})


# Milestone 2: LoRA injection + training loop
def _tiny_model():
    pytest.importorskip("peft")
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=2, n_head=2))


def test_milestone_2_lora_freezes_base_and_loss_goes_down():
    from app.lora import add_lora, count_params, make_batches, train_loop
    model = add_lora(_tiny_model())
    trainable, total = count_params(model)
    assert 0 < trainable < total * 0.2
    enc = [{"input_ids": [1, 2, 3, 4, 5, 6], "labels": [-100, -100, 3, 4, 5, 6]}] * 8
    history = train_loop(model, make_batches(enc, 4, pad_id=0), epochs=25, lr=1e-2)
    assert history[-1] < history[0]


# Milestone 3: UI helpers
def test_milestone_3_clean_commentary_keeps_one_line_and_page_compiles():
    raw = " What a goal!\nEvent: Football | 9'" + EOS + "junk"
    assert clean_commentary(raw) == "What a goal!"
    import py_compile
    py_compile.compile("app/main.py", doraise=True)


# Milestone 4: evaluation metrics
def test_milestone_4_relevance_and_form_checks():
    assert relevance_score(["Rivera", "Red Lions"], "Rivera scores!") == 0.5
    assert relevance_score(["Rivera"], "RIVERA!") == 1.0
    assert relevance_score([], "x") == 0.0
    assert is_well_formed("Great strike!") and not is_well_formed("") and not is_well_formed("no end")


# Milestone 5: merging must not change the model's outputs
def test_milestone_5_merged_model_matches_unmerged():
    import torch
    from app.lora import add_lora
    model = add_lora(_tiny_model())
    for n, p in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.5)  # B starts at 0, so make the adapter actually do something
    model.eval()
    x = torch.tensor([[1, 2, 3, 4]])
    before = model(x).logits
    merged = model.merge_and_unload()
    assert not any("lora" in n for n, _ in merged.named_parameters())
    assert torch.allclose(before, merged(x).logits, atol=1e-4)
