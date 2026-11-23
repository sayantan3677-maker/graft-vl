import random

from vlm.hybrid.formats import STANDARD_BRIEF, STANDARD_LETTER, normalize_sample


def _s(question, answer):
    return {"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
                         {"role": "assistant", "content": answer}]}


def _qa(s):
    return s["messages"][0]["content"][-1]["text"], s["messages"][1]["content"]


def test_brief_answer_period_stripped():
    q, a = _qa(normalize_sample(_s("What is the name?\nGive a very brief answer.", "WASHINGTON UNIVERSITY."), random.Random(0), 0.0))
    assert a == "WASHINGTON UNIVERSITY" and q.endswith("Give a very brief answer.")


def test_abbreviations_and_long_answers_kept():
    assert _qa(normalize_sample(_s("Where?\nKeep it brief.", "U.S."), random.Random(0), 0.0))[1] == "U.S."
    assert _qa(normalize_sample(_s("Who?\nKeep it brief.", "Acme Inc."), random.Random(0), 0.0))[1] == "Acme Inc."
    long = "The chart shows a steady rise in revenue over the decade."
    assert _qa(normalize_sample(_s("Why?\nKeep it brief.", long), random.Random(0), 0.0))[1] == long


def test_letter_mcq_bare_letter_and_standard_instruction():
    q, a = _qa(normalize_sample(_s("Question: color?\nChoices:\nA. Blue.\nB. Red.\nAnswer with the letter.", "Answer: B"),
                                random.Random(0), 1.0))
    assert a == "B" and q.endswith(STANDARD_LETTER) and "Choices:\nA. Blue." in q


def test_brief_instruction_standardized():
    q, a = _qa(normalize_sample(_s("How many?\nGive a very brief answer.", "2."), random.Random(0), 1.0))
    assert a == "2" and q == "How many?\n" + STANDARD_BRIEF


def test_untouched_when_no_known_instruction():
    s = _s("Answer: format please.\nOptions: a, b.", "Answer: serving.\nRationale: x.")
    assert _qa(normalize_sample(s, random.Random(0), 1.0)) == ("Answer: format please.\nOptions: a, b.", "Answer: serving.\nRationale: x.")


def test_pick_turn_single_turn_shape():
    from vlm.hybrid.formats import pick_turn

    s = {"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q1"}]},
                      {"role": "assistant", "content": "A1"},
                      {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}], "meta": {"n_turns": 2}}
    seen = set()
    for seed in range(20):
        out = pick_turn(s, random.Random(seed))
        assert len(out["messages"]) == 2 and out["messages"][0]["content"][0] == {"type": "image"}
        seen.add((out["messages"][0]["content"][1]["text"], out["messages"][1]["content"]))
    assert seen == {("Q1", "A1"), ("Q2", "A2")}


def test_yes_no_and_unprompted_qa():
    assert _qa(normalize_sample(_s("Is it red?\nAnswer yes or no.", "No."), random.Random(0), 0.0))[1] == "No"
    assert _qa(normalize_sample(_s("Who finished next?", "Charles Zwolsman."), random.Random(0), 0.0))[1] == "Charles Zwolsman."
    assert _qa(normalize_sample(_s("Who finished next?", "Charles Zwolsman."), random.Random(0), 0.0, strip_unprompted=True))[1] == "Charles Zwolsman"


def test_rationale_answer_line_removed():
    why = "The response time for the Basic tier is listed in the table; this value is 36 hours."
    s = _s("What is the response time for the Basic tier?", why + "\nAnswer: 36 hours")
    q, a = _qa(normalize_sample(s, random.Random(0), 0.0, rationale=True))
    assert a == why and q == "What is the response time for the Basic tier?"


def test_rationale_answer_not_stated_gets_closing_sentence():
    s = _s("Which option is correct?", "Angle E = 22.5 degrees, so the first option holds.\nAnswer:A")
    assert _qa(normalize_sample(s, random.Random(0), 0.0, rationale=True))[1].endswith("So the answer is A.")


def test_rationale_answer_brief_instruction_variant():
    s = _s("When did the first call occur?", "It is the first log entry.\nAnswer: May 15, 2023.")
    q, a = _qa(normalize_sample(s, random.Random(0), 1.0, rationale=True))
    assert a == "May 15, 2023" and q == "When did the first call occur?\n" + STANDARD_BRIEF
