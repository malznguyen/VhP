"""Gói kết quả một câu hỏi (dùng chung cho build_index và UI)."""
from __future__ import annotations

from itertools import product

VARIANT_KS = (3, 5)
MODES = ("strict", "partial")


def answer_key(side: str, k: int, inherit: bool, mode: str) -> str:
    if side == "baseline":
        inherit = False
    return f"{side}|k{k}|inh{int(inherit)}|{mode}"


def passages_for(result: dict, side: str, k: int, inherit: bool) -> list:
    if side == "baseline":
        return result["baseline"]["passages"][:k]
    v = result["vihippo"]
    src = v.get("passages_inherit_top10" if inherit else "passages_top10") or v.get("passages_inherit" if inherit else "passages")
    return src[:k]


def run_question(eng, question: str) -> dict:
    """Truy xuất một lần (lấy top-10) để mọi biến thể top-3/top-5 chỉ là lát cắt."""
    b = eng.baseline(question, top_k=10)
    v = eng.explain(question, top_k=10)
    return {"question": question, "baseline": b, "vihippo": v, "answers": {}}


def get_answer(eng, result: dict, side: str, k: int, inherit: bool, mode: str) -> dict | None:
    key = answer_key(side, k, inherit, mode)
    if key in result["answers"]:
        return result["answers"][key]
    if eng is None:
        return None
    ps = passages_for(result, side, k, inherit)
    if side == "baseline":
        ps = [{**p, "note": None} for p in ps]
    ans = eng.answer(result["question"], ps, mode=mode)
    result["answers"][key] = ans
    return ans


def precompute_all(eng, result: dict):
    for side, k, inh, mode in product(("baseline", "vihippo"), VARIANT_KS, (False, True), MODES):
        if side == "baseline" and inh:
            continue
        get_answer(eng, result, side, k, inh, mode)
    result["no_rag"] = eng.answer_without_docs(result["question"])
