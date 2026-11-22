"""Short-answer matching shared by trace verification and QA evaluation."""

import re

def norm(s: str) -> str:
    s = s.lower().strip().rstrip(".")
    s = re.sub(r"^(the answer is|answer:)\s*", "", s)
    return re.sub(r"[\s\"'`*]+", " ", s).strip()


RELAXED_NUMERIC = {"chartqa", "tat_qa", "infographic_vqa"}  # ChartQA convention: 5% tolerance on numbers


def answers_match(pred: str, gold: str, relaxed: bool = False) -> bool:
    p, g = norm(pred), norm(gold)
    if not p:
        return False
    if p == g:
        return True
    to_num = lambda x: float(x.replace(",", "").replace("%", "").replace("$", ""))
    try:
        pn, gn = to_num(p), to_num(g)
        if not relaxed:  # counts, years, exact values
            return pn == gn
        return abs(pn - gn) <= 0.05 * abs(gn) if gn != 0 else pn == 0
    except ValueError:
        pass
    m = re.match(r"^\(?([a-j])\)?(\W|$)", p)  # multiple-choice letter
    if len(g) == 1 and m:
        return m.group(1) == g
    return len(g) >= 3 and (g in p.split(" ") or (len(p) <= 3 * len(g) and g in p))
