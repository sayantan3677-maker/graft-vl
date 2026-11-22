"""Text-regression task loaders and scorers (roadmap §7.1–7.2).

Each task yields items {"id", "prompt", "gold", ...} and scores one completion.
Answers are extracted from the final answer section (after </think> when present).

Tasks: mmlu_pro, gsm8k, math500, aime24, aime25, gpqa_diamond, ifeval
"""

import random
import re
from dataclasses import dataclass
from typing import Callable

LETTERS = "ABCDEFGHIJ"


def strip_thinking(text: str) -> str:
    return text.split("</think>", 1)[1] if "</think>" in text else text


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        depth += text[k] == "{"
        depth -= text[k] == "}"
        if depth == 0:
            return text[j + 1: k]
    return None


def math_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        from math_verify import parse, verify

        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:
        norm = lambda s: re.sub(r"\s|\\!|\\,|\$|\\left|\\right|\\text\{\s*\}", "", s)
        return norm(pred) == norm(gold)


def extract_choice(text: str, n_options: int) -> str | None:
    valid = LETTERS[:n_options]
    for pat in [rf"answer is \(?([{valid}])\)?", rf"[Aa]nswer:\s*\(?([{valid}])\)?", rf"\\boxed\{{\(?([{valid}])\)?\}}"]:
        m = re.findall(pat, text)
        if m:
            return m[-1]
    m = re.findall(rf"\b([{valid}])\b", text[-200:])
    return m[-1] if m else None


@dataclass
class Task:
    name: str
    load: Callable[[int | None, int], list[dict]]
    score: Callable[[dict, str], float]


# --------------------------------------------------------------------------- loaders


def _subset(items: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or limit >= len(items):
        return items
    rng = random.Random(seed)
    return sorted(rng.sample(items, limit), key=lambda x: str(x["id"]))


def load_mmlu_pro(limit, seed):
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    items = []
    for r in ds:
        opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(r["options"]))
        prompt = (f"The following is a multiple choice question about {r['category']}.\n\n{r['question']}\n\n{opts}\n\n"
                  'Answer with the letter of the correct option. End with "The answer is (X)".')
        items.append({"id": r["question_id"], "prompt": prompt, "gold": r["answer"], "n_options": len(r["options"])})
    return _subset(items, limit, seed)


def load_gsm8k(limit, seed):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = [{"id": i, "prompt": f"{r['question']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
              "gold": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in enumerate(ds)]
    return _subset(items, limit, seed)


def _math_items(rows, id_key, q_key, a_key):
    return [{"id": r[id_key], "prompt": f"{r[q_key]}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
             "gold": str(r[a_key])} for r in rows]


def load_math500(limit, seed):
    from datasets import load_dataset

    return _subset(_math_items(load_dataset("HuggingFaceH4/MATH-500", split="test"), "unique_id", "problem", "answer"), limit, seed)


def load_aime24(limit, seed):
    from datasets import load_dataset

    return _subset(_math_items(load_dataset("HuggingFaceH4/aime_2024", split="train"), "id", "problem", "answer"), limit, seed)


def load_aime25(limit, seed):
    from datasets import load_dataset

    ds = load_dataset("yentinglin/aime_2025", split="train")
    return _subset(_math_items(ds, "id", "problem", "answer"), limit, seed)


def load_gpqa_diamond(limit, seed):
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")  # gated: accept terms on the Hub
    items = []
    for i, r in enumerate(ds):
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = random.Random(1000 + i).sample(range(4), 4)  # fixed per-question shuffle
        shown = [opts[j] for j in order]
        gold = LETTERS[order.index(0)]
        body = "\n".join(f"({LETTERS[k]}) {o.strip()}" for k, o in enumerate(shown))
        prompt = f"{r['Question'].strip()}\n\n{body}\n\nAnswer with the letter of the correct option. End with \"The answer is (X)\"."
        items.append({"id": i, "prompt": prompt, "gold": gold, "n_options": 4})
    return _subset(items, limit, seed)


def load_ifeval(limit, seed):
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    items = [{"id": r["key"], "prompt": r["prompt"], "gold": None, "doc": dict(r)} for r in ds]
    return _subset(items, limit, seed)


# --------------------------------------------------------------------------- scorers


def score_choice(item, completion):
    return float(extract_choice(strip_thinking(completion), item.get("n_options", 10)) == item["gold"])


def score_math(item, completion):
    return float(math_equal(last_boxed(strip_thinking(completion)), item["gold"]))


def score_gsm8k(item, completion):
    ans = strip_thinking(completion)
    pred = last_boxed(ans)
    if pred is None:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", ans)
        pred = nums[-1] if nums else None
    if pred is None:
        return 0.0
    pred = pred.replace(",", "").replace("$", "").strip()
    try:
        return float(abs(float(pred) - float(item["gold"])) < 1e-6)
    except ValueError:
        return float(math_equal(pred, item["gold"]))


def score_ifeval(item, completion):
    """Prompt-level strict accuracy (all instructions followed)."""
    from lm_eval.tasks.ifeval.utils import process_results

    res = process_results(item["doc"], [strip_thinking(completion).strip()])
    return float(res["prompt_level_strict_acc"])


TASKS: dict[str, Task] = {
    "mmlu_pro": Task("mmlu_pro", load_mmlu_pro, score_choice),
    "gsm8k": Task("gsm8k", load_gsm8k, score_gsm8k),
    "math500": Task("math500", load_math500, score_math),
    "aime24": Task("aime24", load_aime24, score_math),
    "aime25": Task("aime25", load_aime25, score_math),
    "gpqa_diamond": Task("gpqa_diamond", load_gpqa_diamond, score_choice),
    "ifeval": Task("ifeval", load_ifeval, score_ifeval),
}
