"""Lập chỉ mục + tính sẵn kết quả cho UI.

    python build_index.py              # lập chỉ mục (OpenIE, đồ thị) + cache 3 câu mẫu
    python build_index.py --baseline   # chỉ chạy RAG thường cho các câu mẫu và in kết quả
    python build_index.py --rebuild    # xoá chỉ mục cũ, làm lại từ đầu (gọi lại LLM)

Mọi lời gọi LLM được HippoRAG cache trong outputs/hipporag/llm_cache (sqlite);
kết quả cho UI được ghi ra cache/*.json để `python app.py --offline` chạy không cần mạng.
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.stdout.reconfigure(encoding="utf-8")

from vihippo import config as C
from vihippo.corpus import load_corpus


def get_engine():
    from vihippo.engine import ViHippoRAG
    corpus = load_corpus(C.CORPUS_DIR, core_files=C.CORE_FILES, reference_date=C.REFERENCE_DATE)
    return corpus, ViHippoRAG(corpus)


def dump(name, obj):
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (C.CACHE_DIR / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def ablation(eng, question, gold):
    """Gỡ dần từng thành phần để xem phần nào làm thay đổi thứ hạng đoạn đích (không gọi LLM mới)."""
    import numpy as np
    if not gold:
        return []
    full = eng.graph.copy()
    sup = set(eng.superseded_idx)
    lo, hi = eng.doc_vertex_range
    rows = []
    for name, doc_layer, penalty, inherit in [
        ("HippoRAG 2 gốc", False, False, False),
        ("+ nút văn bản & cạnh quan hệ", True, False, False),
        ("+ giảm điểm đoạn đã bị sửa (đặc tả)", True, True, False),
        ("+ đoạn sửa đổi kế thừa điểm (tuỳ chọn)", True, True, True),
    ]:
        eng.graph = full.copy()
        if not doc_layer:
            eng.graph.delete_vertices(list(range(lo, hi)))
        eng.superseded_idx = sup if penalty else set()
        fs = eng.get_fact_scores(question)
        ti, tf, _ = eng.rerank_facts(question, fs)
        if tf:
            eng.graph_search_with_fact_entities(question, eng.global_config.linking_top_k, fs, tf, ti,
                                                eng.global_config.passage_node_weight)
            raw = eng._last_raw_doc_scores
        else:
            ids, sc = eng.dense_passage_retrieval(question)
            raw = np.zeros(len(eng.passage_node_keys)); raw[ids] = sc
        order, _ = eng.apply_validity(raw, inherit=inherit)
        pids = [eng.key_to_pid[eng.passage_node_keys[i]] for i in order]
        row = {"name": name, "gold_ranks": {g: pids.index(g) + 1 for g in gold},
               "top": [eng.corpus.passages[x].citation for x in pids[:C.TOP_K]]}
        if not doc_layer:  # HippoRAG 2 gốc: trả lời luôn bằng top-k của nó (không nhãn hiệu lực)
            ps = [{"citation": eng.corpus.passages[x].citation, "text": eng.corpus.passages[x].text, "note": None}
                  for x in pids[:C.TOP_K]]
            row["answer"] = eng.answer(question, ps, mode="strict")
        rows.append(row)
    eng.graph = full
    eng.superseded_idx = sup
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)

    if args.rebuild:
        shutil.rmtree(C.OUTPUT_DIR, ignore_errors=True)
        shutil.rmtree(C.CACHE_DIR, ignore_errors=True)

    corpus, eng = get_engine()
    t0 = time.time()
    eng.index_corpus()
    print(f"\n== Lập chỉ mục xong sau {time.time() - t0:.0f}s: {len(corpus.passages)} đoạn, {len(corpus.docs)} văn bản")

    if args.baseline:
        for q in C.SAMPLE_QUESTIONS:
            b = eng.baseline(q["question"])
            ans = eng.answer(q["question"], b["passages"])
            print(f"\n### {q['question']}\n  kỳ vọng: {q['expected']}")
            for p in b["passages"]:
                print(f"  - {p['score']:.3f} [{p['citation']}] {p['text'][:110]}")
            print("  => RAG thường trả lời:", ans["text"].replace("\n", " "))
        return

    dump("corpus.json", corpus.to_json())
    dump("index_view.json", eng.index_view())
    from vihippo.pipeline import run_question, precompute_all, get_answer
    samples = {}
    for q in C.SAMPLE_QUESTIONS:
        t = time.time()
        res = run_question(eng, q["question"])
        precompute_all(eng, res)
        res["meta"] = q
        res["ablation"] = ablation(eng, q["question"], q.get("gold", []))
        res["amend_share"] = eng.amend_edge_share()
        samples[q["id"]] = res
        print(f"\n### {q['question']}  ({time.time() - t:.1f}s)")
        for side, inh in (("baseline", False), ("vihippo", False), ("vihippo", True)):
            for k in (3, 5):
                for mode in ("strict", "partial"):
                    a = get_answer(None, res, side, k, inh, mode)
                    tag = f"{side}{'+kế thừa' if inh else ''} top{k} {mode}"
                    print(f"  {tag:34s} → {a['text'][:150].replace(chr(10), ' ')}")
        for row in res["ablation"]:
            print(f"  [ablation] {row['name']}: {row['gold_ranks']}")
            if row.get("answer"):
                print(f"  [HippoRAG 2 gốc top{C.TOP_K}] {[c[-26:] for c in row['top']]}")
                print(f"      → {row['answer']['text'][:400]}")
    dump("samples.json", samples)
    print("\nĐã ghi cache/ (corpus.json, index_view.json, samples.json)")


if __name__ == "__main__":
    main()
