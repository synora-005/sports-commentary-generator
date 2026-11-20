"""Smoke checks: the pieces import and the app file is valid."""
import py_compile

from app import core


def test_the_page_file_is_valid_python():
    py_compile.compile("app/main.py", doraise=True)


def test_training_script_is_valid_python():
    py_compile.compile("scripts/train.py", doraise=True)
    py_compile.compile("scripts/evaluate.py", doraise=True)


def test_dataset_file_matches_the_generator_format():
    rows = core.load_jsonl("data/commentary.jsonl")
    assert len(rows) >= 100
    assert {"event", "commentary", "keywords"} <= set(rows[0])