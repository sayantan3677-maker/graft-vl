"""Answer-format normalization for short-answer and multiple-choice training data (S3, roadmap §16 2026-11-20).

FineVision targets contradict their own instructions: "Give a very brief answer." → "WASHINGTON UNIVERSITY." and
"Answer with the letter." → "Answer: B". The S2 model copied both habits (39% of 4B ChartQA answers end in '.', AI2D
answers are "Answer: B"), which breaks strict scorers and downstream parsers. This rewrites the final assistant turn so
it does exactly what the prompt asks, and diversifies the instruction toward the phrasings product/eval prompts use.
"""

import random
import re

BRIEF_INSTRUCTIONS = {
    "Give a very brief answer.", "Answer the question with a short phrase.", "Provide a short and direct response.",
    "Your answer should be very brief.", "Short answer required.", "Keep it brief.", "Keep it short and to the point.",
    "Answer briefly.", "Quick response, please.", "Offer a very short reply.", "Be succinct.",
    "Your response must be concise.", "Write a very short answer.", "Your answer should be compact.",
    "Concise answer only.", "Ensure brevity in your answer.", "Offer a terse response.", "Make the answer very short.",
    "Provide a succinct answer.", "Answer the question using a single word or phrase.",
}
LETTER_INSTRUCTIONS = {"Answer with the letter."}
STANDARD_BRIEF = "Answer the question using a single word or phrase."
STANDARD_LETTER = "Answer with the option's letter from the given choices directly."
_ANSWER_LETTER = re.compile(r"^\s*answer\s*:\s*\(?([A-Z])\)?\.?\s*$", re.I)
_ABBREV_END = re.compile(r"(\b[A-Za-z]\.|\b(?:etc|Inc|Ltd|Co|Corp|Jr|Sr|St|Dr|Mr|Mrs|Ms|vs)\.)$")


def _strip_period(ans: str) -> str:
    """Drop one trailing '.' from a short single-line answer unless it ends an abbreviation (U.S., Inc., etc.)."""
    a = ans.strip()
    if "\n" in a or len(a.split()) > 6 or not a.endswith(".") or a.endswith("..") or _ABBREV_END.search(a):
        return a
    return a[:-1].rstrip()


def _set_instruction(text: str, old: str, new: str) -> str:
    head, _, _ = text.rstrip().rpartition(old)
    return head + new


def normalize_sample(sample: dict, rng: random.Random, p_standard: float = 0.35, strip_unprompted: bool = False) -> dict:
    """Rewrite the final user/assistant pair of a single-turn short-answer or letter-MCQ sample in place.
    strip_unprompted: QA sources whose questions carry no format instruction (table/doc QA): also drop the trailing
    '.' of short answers. Not for transcription sources, where the period is part of the text."""
    msgs = sample["messages"]
    if len(msgs) < 2 or msgs[-1]["role"] != "assistant" or msgs[-1].get("reasoning"):
        return sample
    user = msgs[-2]
    parts = user["content"] if isinstance(user["content"], list) else None
    text = parts[-1]["text"] if parts and parts[-1].get("type") == "text" else (user["content"] if isinstance(user["content"], str) else None)
    if text is None:
        return sample
    last = text.strip().split("\n")[-1].strip()
    ans = msgs[-1]["content"]
    new_text = text
    if last in LETTER_INSTRUCTIONS:
        m = _ANSWER_LETTER.match(ans)
        if m:
            ans = m.group(1).upper()
        if rng.random() < p_standard:
            new_text = _set_instruction(text, last, STANDARD_LETTER)
    elif last in BRIEF_INSTRUCTIONS:
        ans = _strip_period(ans)
        if rng.random() < p_standard:
            new_text = _set_instruction(text, last, STANDARD_BRIEF)
    elif last == "Answer yes or no." or strip_unprompted:
        ans = _strip_period(ans)
    else:
        return sample
    msgs[-1] = {**msgs[-1], "content": ans}
    if new_text is not text:
        if parts:
            parts[-1] = {**parts[-1], "text": new_text}
        else:
            msgs[-2] = {**user, "content": new_text}
    return sample


def pick_turn(sample: dict, rng: random.Random) -> dict:
    """Multi-turn QA sample (one image, several independent questions; build_shards.py --all-turns) → a single-turn
    sample for one randomly drawn question, with the image placed before it exactly as in single-turn data."""
    msgs = sample["messages"]
    if len(msgs) <= 2:
        return sample
    k = rng.randrange(len(msgs) // 2)
    if k == 0:
        return {**sample, "messages": msgs[:2]}
    q = msgs[2 * k]["content"]
    q = q if isinstance(q, str) else "".join(p.get("text", "") for p in q if p.get("type") == "text")
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}
    return {**sample, "messages": [user, msgs[2 * k + 1]]}
