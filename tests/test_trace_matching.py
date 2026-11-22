import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("gen_traces", Path(__file__).resolve().parents[1] / "scripts" / "gen_traces.py")
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)


def test_answers_match():
    assert gt.answers_match("42", "42")
    assert not gt.answers_match("41", "42")                      # counts must be exact
    assert gt.answers_match("The answer is 41.5", "42", relaxed=True)  # charts: 5% tolerance
    assert not gt.answers_match("38", "42", relaxed=True)
    assert gt.answers_match("B", "b")
    assert gt.answers_match("(C) the lungs", "c")
    assert gt.answers_match("Samsung Galaxy", "samsung")
    assert not gt.answers_match("", "yes")
    assert not gt.answers_match("no", "yes")


def test_forbidden_phrases():
    assert gt.FORBIDDEN.search("According to the description, the bar is tall.")
    assert gt.FORBIDDEN.search("The visual facts say 12.")
    assert gt.FORBIDDEN.search("the OCR text shows")
    assert not gt.FORBIDDEN.search("Looking at the chart, the tallest bar is 2019.")
