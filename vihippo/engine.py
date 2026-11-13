"""ViHippo: lõi HippoRAG 2 (repo OSU-NLP-Group) + prompt tiếng Việt + lớp văn bản/quan hệ.

Không sửa mã thư viện. Mọi thay đổi nằm trong lớp con `ViHippoRAG`:
  * thay template OpenIE/NER-câu-hỏi/bộ lọc bằng bản tiếng Việt;
  * sau khi thư viện dựng xong đồ thị (cụm từ + đoạn văn + cạnh đồng nghĩa), gắn thêm
    nút văn bản và cạnh có trọng số (sửa đổi > dẫn chiếu > căn cứ) vào đồ thị trong bộ nhớ;
  * giảm điểm đoạn đã bị sửa đổi sau PPR (không xoá);
  * ghi lại các bước trung gian để hiển thị (a)–(e).
"""
from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from datetime import date
from typing import Dict, List, Optional

import numpy as np

from hipporag import HippoRAG
from hipporag.rerank import DSPyFilter
from hipporag.utils.config_utils import BaseConfig

from . import config as C
from .corpus import Corpus, RELATION_LABELS
from . import prompts_vi as P


# ------------------------------------------------------------------ bộ lọc VI
class ViFilter(DSPyFilter):
    """Bộ lọc bộ ba (recognition memory) của HippoRAG 2 với prompt tiếng Việt + lý do."""

    def __init__(self, hipporag):
        self.one_input_template = P.FILTER_INPUT
        self.one_output_template = P.FILTER_OUTPUT
        self.message_template = self._make_vi_template()
        self.llm_infer_fn = hipporag.extraction_llm.infer
        self.default_gen_kwargs = {"max_new_tokens": 4096}
        self.last_response: str = ""

    def _make_vi_template(self):
        msgs = [{"role": "system", "content": P.FILTER_DSPY["prog"]["system"]}]
        for d in P.FILTER_DSPY["prog"]["demos"]:
            msgs.append({"role": "user", "content": self.one_input_template.format(
                question=d["question"], fact_before_filter=d["fact_before_filter"])})
            msgs.append({"role": "assistant", "content": self.one_output_template.format(
                fact_after_filter=d["fact_after_filter"], reason=d["reason"] or "(không có)")})
        return msgs

    def llm_call(self, question, fact_before_filter):
        resp = super().llm_call(question, fact_before_filter)
        self.last_response = resp if isinstance(resp, str) else str(resp)
        return resp

    def last_reasons(self) -> str:
        m = re.search(r"\[\[ ## reason ## \]\](.*?)(\[\[ ## completed ## \]\]|$)", self.last_response, re.S)
        return m.group(1).strip() if m else ""


def make_config() -> BaseConfig:
    C.load_env()
    cfg = BaseConfig()
    cfg.save_dir = str(C.OUTPUT_DIR)
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    cfg.llm_name = deployment
    cfg.azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    cfg.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or None
    cfg.azure_chat_deployment = deployment
    cfg.llm_api_key_env = "AZURE_OPENAI_API_KEY"
    t = os.getenv("LLM_TEMPERATURE", "")
    cfg.temperature = float(t) if t else None  # model suy luận (gpt-5.x) không nhận temperature=0
    cfg.seed = None
    cfg.max_new_tokens = None
    cfg.openie_ner_max_tokens = 8192
    cfg.openie_triple_max_tokens = 12000
    cfg.openie_max_workers = 8
    cfg.embedding_model_name = f"Transformers/{C.EMBEDDING_MODEL}"
    cfg.embedding_batch_size = 16
    cfg.retrieval_top_k = 20
    cfg.qa_top_k = C.TOP_K
    cfg.linking_top_k = 5
    cfg.damping = 0.5
    cfg.passage_node_weight = 0.05
    cfg.synonymy_edge_sim_threshold = float(os.getenv("VIHIPPO_SYN_THRESHOLD", "0.8"))
    return cfg


class ViHippoRAG(HippoRAG):
    def __init__(self, corpus: Corpus, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config=global_config or make_config(), index_identity="vihippo-vi-prompts-v1")
        self.corpus = corpus
        for mgr in (self.openie.prompt_template_manager, self.prompt_template_manager):
            mgr.templates["ner"] = P.to_templates(P.NER_TEMPLATE)
            mgr.templates["triple_extraction"] = P.to_templates(P.TRIPLE_TEMPLATE)
            mgr.templates["ner_query"] = P.to_templates(P.QUERY_NER_TEMPLATE)
        self.rerank_filter = ViFilter(self)
        self._doc_vertex_names: List[str] = []
        self._last_reset = None
        self._last_ppr = None
        self._last_raw_doc_scores = None

    # ----------------------------------------------------------- lập chỉ mục
    def index_corpus(self):
        from hipporag.preprocessing import Chunk
        chunks = [Chunk(content=p.content, metadata={"pid": p.pid}) for p in self.corpus.passage_list()]
        self.index(chunks)

    # ------------------------------------------------- lớp văn bản trên đồ thị
    def prepare_retrieval_objects(self):
        if self._doc_vertex_names:  # gỡ lớp văn bản cũ để thư viện kiểm tra nhất quán
            self.graph.delete_vertices([v.index for v in self.graph.vs if v["name"] in set(self._doc_vertex_names)])
            self._doc_vertex_names = []
        super().prepare_retrieval_objects()
        self._attach_document_layer()

    def _attach_document_layer(self):
        g = self.graph
        text_to_key = self.chunk_embedding_store.text_to_hash_id
        self.pid_to_key = {pid: text_to_key[p.content] for pid, p in self.corpus.passages.items() if p.content in text_to_key}
        self.key_to_pid = {v: k for k, v in self.pid_to_key.items()}
        doc_names = [f"doc::{d}" for d in self.corpus.docs]
        start = g.vcount()
        g.add_vertices(len(doc_names), attributes={
            "name": doc_names,
            "content": [self.corpus.docs[d].short for d in self.corpus.docs],
            "hash_id": doc_names,
        })
        self._doc_vertex_names = doc_names
        name_to_idx = {n: i for i, n in enumerate(g.vs["name"])}
        edges, weights, kinds = [], [], []

        def add(a, b, w, kind):
            edges.append((name_to_idx[a], name_to_idx[b]))
            weights.append(w)
            kinds.append(kind)

        for pid, key in self.pid_to_key.items():
            add(key, f"doc::{self.corpus.passages[pid].doc_id}", C.BELONG_WEIGHT, "thuoc_van_ban")
        for r in self.corpus.relations:
            src = self.pid_to_key.get(r.src_pid) if r.src_pid else f"doc::{r.src_doc}"
            if r.dst_pid and r.dst_pid in self.pid_to_key:
                dst = self.pid_to_key[r.dst_pid]
            else:
                dst = f"doc::{r.dst_doc}"
            add(src, dst, r.weight, r.kind)
            if r.src_pid and r.dst_pid:  # thêm cạnh mức văn bản cho quan hệ mức khoản
                add(f"doc::{r.src_doc}", f"doc::{r.dst_doc}", r.weight, r.kind)
        g.add_edges(edges, attributes={"weight": weights, "edge_kind": kinds})
        self.doc_vertex_range = (start, g.vcount())
        pos = {key: i for i, key in enumerate(self.passage_node_keys)}
        self.superseded_idx = set()
        self.amend_pairs = []  # (chỉ số đoạn bị sửa, chỉ số đoạn sửa đổi)
        for pid, p in self.corpus.passages.items():
            if p.superseded_by and pid in self.pid_to_key and p.superseded_by in self.pid_to_key:
                a, b = pos[self.pid_to_key[pid]], pos[self.pid_to_key[p.superseded_by]]
                self.superseded_idx.add(a)
                self.amend_pairs.append((a, b))

    # ------------------------------------------ PPR + giảm điểm đoạn đã bị sửa
    def run_ppr(self, reset_prob: np.ndarray, damping: float = 0.5):
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        n = self.graph.vcount()
        if len(reset_prob) < n:  # vector hạt giống của thư viện chưa tính nút văn bản
            reset_prob = np.concatenate([reset_prob, np.zeros(n - len(reset_prob))])
        ppr = np.array(self.graph.personalized_pagerank(
            vertices=range(n), damping=damping, directed=False, weights="weight",
            reset=reset_prob, implementation="prpack"))
        self._last_reset, self._last_ppr = reset_prob, ppr
        raw = np.array([ppr[i] for i in self.passage_node_idxs])
        self._last_raw_doc_scores = raw.copy()
        return self.apply_validity(raw, inherit=False)

    def apply_validity(self, raw: np.ndarray, inherit: bool = False):
        """Quy tắc hiệu lực sau PPR.
        - luôn: đoạn đã bị sửa giữ nguyên trong kết quả nhưng điểm × SUPERSEDED_FACTOR;
        - inherit=True (tuỳ chọn, mặc định TẮT): đoạn sửa đổi nhận ít nhất điểm gốc của đoạn mà nó sửa.
        """
        scores = np.asarray(raw, dtype=float).copy()
        for i in self.superseded_idx:
            scores[i] *= C.SUPERSEDED_FACTOR
        if inherit:
            for a, b in self.amend_pairs:
                scores[b] = max(scores[b], raw[a])
        order = np.argsort(scores)[::-1]
        return order, scores[order]

    # ------------------------------------------------------------ hiển thị
    def _node_info(self, idx: int) -> dict:
        v = self.graph.vs[idx]
        name = v["name"]
        if name.startswith("doc::"):
            return {"type": "văn bản", "label": v["content"], "name": name}
        if name.startswith("chunk-"):
            pid = self.key_to_pid.get(name)
            p = self.corpus.passages.get(pid) if pid else None
            return {"type": "đoạn văn", "label": p.citation if p else v["content"][:60], "name": name, "pid": pid}
        return {"type": "cụm từ", "label": v["content"], "name": name}

    def _passage_row(self, passage_idx: int, score: float, raw: Optional[float] = None) -> dict:
        key = self.passage_node_keys[passage_idx]
        pid = self.key_to_pid.get(key)
        p = self.corpus.passages[pid]
        return {"pid": pid, "citation": p.citation, "text": p.text, "content": p.content,
                "doc_id": p.doc_id, "score": float(score), "raw_score": None if raw is None else float(raw),
                "note": p.superseded_note}

    def ppr_iterations(self, reset: np.ndarray, damping: float, steps: int = 12, keep: int = 25) -> dict:
        """Lặp lũy thừa (power iteration) để minh hoạ điểm lan theo vòng; kết quả cuối ≈ igraph PPR."""
        g = self.graph
        n = g.vcount()
        W = np.zeros((n, n))
        for e, w in zip(g.es, g.es["weight"]):
            W[e.source, e.target] += w or 0.0
            W[e.target, e.source] += w or 0.0
        deg = W.sum(1)
        r = reset / reset.sum()
        x = r.copy()
        hist = [x.copy()]
        for _ in range(steps):
            walk = np.zeros(n)
            nz = deg > 0
            walk += (x[nz] / deg[nz]) @ W[nz]
            walk += x[~nz].sum() * r  # nút cô lập: quay về hạt giống
            x = (1 - damping) * r + damping * walk
            hist.append(x.copy())
        final = self._last_ppr if self._last_ppr is not None else x
        top = list(np.argsort(final)[::-1][:keep])
        return {
            "nodes": [self._node_info(int(i)) for i in top],
            "node_idx": [int(i) for i in top],
            "scores": [[float(h[i]) for i in top] for h in hist],
        }

    def subgraph(self, node_idx: List[int], highlight: Dict[int, float] | None = None, extra_neighbors: bool = False) -> dict:
        g = self.graph
        keep = set(node_idx)
        if extra_neighbors:
            for i in list(node_idx):
                keep.update(g.neighbors(i))
        nodes = []
        for i in keep:
            info = self._node_info(i)
            info["id"] = int(i)
            info["seed"] = float(highlight.get(i, 0.0)) if highlight else 0.0
            info["ppr"] = float(self._last_ppr[i]) if self._last_ppr is not None and i < len(self._last_ppr) else 0.0
            nodes.append(info)
        edges = []
        for e in g.es.select(_within=list(keep)):
            kind = e["edge_kind"] if "edge_kind" in g.es.attributes() else None
            edges.append({"s": e.source, "t": e.target, "w": float(e["weight"] or 0), "kind": kind or "fact"})
        return {"nodes": nodes, "edges": edges}

    def amend_edge_share(self) -> list:
        """Với mỗi đoạn bị sửa: cạnh 'sửa đổi' chiếm bao nhiêu phần tổng trọng số cạnh của nút đó."""
        g = self.graph
        rows = []
        for a, b in self.amend_pairs:
            va, vb = self.passage_node_idxs[a], self.passage_node_idxs[b]
            inc = g.incident(va)
            total = sum(g.es[e]["weight"] or 0 for e in inc)
            amend = sum(g.es[e]["weight"] or 0 for e in inc
                        if g.es[e]["edge_kind"] == "sua_doi" and vb in (g.es[e].source, g.es[e].target))
            rows.append({"old": self.corpus.passages[self.key_to_pid[self.passage_node_keys[a]]].citation,
                         "new": self.corpus.passages[self.key_to_pid[self.passage_node_keys[b]]].citation,
                         "n_edges": len(inc), "total_weight": float(total), "amend_weight": float(amend),
                         "share": float(amend / total) if total else 0.0})
        return rows

    # ------------------------------------------------------------- truy xuất
    def explain(self, query: str, top_k: int = C.TOP_K) -> dict:
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()
        self.get_query_embeddings([query])
        fact_scores = self.get_fact_scores(query)
        order = np.argsort(fact_scores)[::-1]
        fact_rows = self.fact_embedding_store.get_rows([self.fact_node_keys[i] for i in order[:10]])
        import ast
        candidates = [{"fact": list(ast.literal_eval(fact_rows[self.fact_node_keys[i]]["content"])),
                       "score": float(fact_scores[i]), "to_llm": rank < self.global_config.linking_top_k}
                      for rank, i in enumerate(order[:10])]

        top_idx, top_facts, _ = self.rerank_facts(query, fact_scores)
        kept = [list(f) for f in top_facts]
        reasons = self.rerank_filter.last_reasons()
        trace = {"query": query, "candidates": candidates, "kept": kept, "filter_reasons": reasons,
                 "filter_raw": self.rerank_filter.last_response}

        if not top_facts:
            ids, scores = self.dense_passage_retrieval(query)
            trace["fallback"] = "Không còn bộ ba nào sau khi lọc → dùng truy xuất vector (như HippoRAG 2 gốc)."
            # vẫn áp hệ số hiệu lực
            raw = np.zeros(len(self.passage_node_keys))
            raw[ids] = scores
            trace.update(seeds=[], top_nodes=[], iterations=None, graph=None)
            for name, inh in (("passages", False), ("passages_inherit", True)):
                o, sc = self.apply_validity(raw, inherit=inh)
                trace[name] = [self._passage_row(int(i), s_, raw[int(i)]) for i, s_ in zip(o[:top_k], sc[:top_k])]
            trace["passages_top10"] = trace["passages"]
            return trace

        ids, scores = self.graph_search_with_fact_entities(
            query=query, link_top_k=self.global_config.linking_top_k, query_fact_scores=fact_scores,
            top_k_facts=top_facts, top_k_fact_indices=top_idx, passage_node_weight=self.global_config.passage_node_weight)

        reset = self._last_reset
        seed_idx = [int(i) for i in np.argsort(reset)[::-1] if reset[i] > 0]
        seeds = []
        for i in seed_idx:
            info = self._node_info(i)
            info["weight"] = float(reset[i])
            seeds.append(info)
        phrase_seeds = [s for s in seeds if s["type"] == "cụm từ"]
        passage_seeds = [s for s in seeds if s["type"] == "đoạn văn"]
        trace["seeds"] = phrase_seeds + passage_seeds[:5]
        trace["n_passage_seeds"] = len(passage_seeds)

        ppr = self._last_ppr
        top_nodes = []
        for i in np.argsort(ppr)[::-1][:20]:
            info = self._node_info(int(i))
            info["ppr"] = float(ppr[i])
            info["is_seed"] = bool(reset[i] > 0 and info["type"] == "cụm từ")
            top_nodes.append(info)
        trace["top_nodes"] = top_nodes

        raw = self._last_raw_doc_scores
        trace["passages"] = [self._passage_row(int(i), s, raw[int(i)]) for i, s in zip(ids[:top_k], scores[:top_k])]
        trace["passages_top10"] = [self._passage_row(int(i), s, raw[int(i)]) for i, s in zip(ids[:10], scores[:10])]
        o2, s2 = self.apply_validity(raw, inherit=True)
        trace["passages_inherit"] = [self._passage_row(int(i), s, raw[int(i)]) for i, s in zip(o2[:top_k], s2[:top_k])]
        trace["passages_inherit_top10"] = [self._passage_row(int(i), s, raw[int(i)]) for i, s in zip(o2[:10], s2[:10])]
        trace["iterations"] = self.ppr_iterations(reset, self.global_config.damping)

        # đồ thị con quanh hạt giống và kết quả để vẽ
        show = [i for i in np.argsort(ppr)[::-1][:28]]
        show += [s for s in seed_idx if self._node_info(s)["type"] == "cụm từ"]
        top_passage_v = [self.passage_node_idxs[int(i)] for i in ids[:C.TOP_K]]
        show += top_passage_v
        trace["graph"] = self.subgraph(list(dict.fromkeys(int(i) for i in show)),
                                       highlight={int(i): float(reset[i]) for i in seed_idx if self._node_info(int(i))["type"] == "cụm từ"})
        trace["graph"]["top_passages"] = top_passage_v
        return trace

    def baseline(self, query: str, top_k: int = C.TOP_K) -> dict:
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()
        self.get_query_embeddings([query])
        ids, scores = self.dense_passage_retrieval(query)
        rows = []
        for i, s in zip(ids[:top_k], scores[:top_k]):
            r = self._passage_row(int(i), s)
            r["note"] = None  # RAG thường không biết quan hệ hiệu lực
            rows.append(r)
        return {"query": query, "passages": rows}

    def answer_without_docs(self, query: str) -> dict:
        msgs = [{"role": "system", "content": P.NO_RAG_SYSTEM}, {"role": "user", "content": query}]
        text, meta, hit = self.qa_llm.infer(messages=msgs, max_new_tokens=4096)
        return {"text": text.strip(), "cache_hit": hit}

    def answer(self, query: str, passages: List[dict], mode: str = "strict") -> dict:
        system = P.QA_SYSTEM if mode == "strict" else P.QA_SYSTEM_PARTIAL
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": P.format_qa_user(query, passages)}]
        text, meta, hit = self.qa_llm.infer(messages=msgs, max_new_tokens=4096)
        return {"text": text.strip(), "cache_hit": hit, "prompt": msgs[1]["content"]}

    # ----------------------------------------------------- dữ liệu trang chỉ mục
    def index_view(self) -> dict:
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()
        openie = {row["idx"]: row for row in (self._openie_info or self.load_existing_openie([])[0])}
        triples = {}
        for key, row in openie.items():
            pid = self.key_to_pid.get(key)
            if pid:
                triples[pid] = {"entities": row.get("extracted_entities", []), "triples": row.get("extracted_triples", [])}
        info = self.get_graph_info()
        n_doc = len(self._doc_vertex_names)
        g = self.graph
        kinds = {}
        for k in g.es["edge_kind"]:
            kinds[k or "?"] = kinds.get(k or "?", 0) + 1
        n_syn = kinds.get("synonym", 0)
        # đồ thị con của các văn bản lõi
        core_pids = [pid for pid, p in self.corpus.passages.items() if self.corpus.docs[p.doc_id].is_core]
        core_v = [self.node_name_to_vertex_idx[self.pid_to_key[pid]] for pid in core_pids if pid in self.pid_to_key]
        doc_v = [g.vs.find(name=f"doc::{d}").index for d, doc in self.corpus.docs.items() if doc.is_core or doc.external]
        ent_v = set()
        for v in core_v:
            for nb in g.neighbors(v):
                if g.vs[nb]["name"].startswith("entity-"):
                    ent_v.add(nb)
        return {
            "triples": triples,
            "stats": {"passages": len(self.passage_node_keys), "phrases": len(self.entity_node_keys),
                      "documents": n_doc, "facts": len(self.fact_node_keys), "edges": g.ecount(),
                      "synonym_edges": n_syn, "edge_kinds": kinds, "library_info": info},
            "core_graph": self.subgraph(core_v + doc_v + sorted(ent_v)),
        }
