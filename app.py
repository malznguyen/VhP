"""ViHippo – bản demo trình bày.

Chạy:  python app.py            (có mạng: hỏi tự do được)
       python app.py --offline  (không mạng: chỉ dùng kết quả đã cache)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _launch():
    offline = "--offline" in sys.argv
    env = dict(os.environ)
    if offline:
        env.update(VIHIPPO_OFFLINE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
           "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    port = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--port=")), None)
    if port:
        cmd += ["--server.port", port]
    raise SystemExit(subprocess.call(cmd, env=env, cwd=str(ROOT)))


if __name__ == "__main__":
    try:
        from streamlit.runtime import exists as _st_running
    except Exception:  # pragma: no cover
        _st_running = lambda: False  # noqa: E731
    if not _st_running():
        _launch()

# ------------------------------------------------------------------ Streamlit
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(ROOT))
from vihippo import config as C  # noqa: E402
from vihippo.pipeline import get_answer, passages_for, run_question  # noqa: E402

OFFLINE = C.is_offline()
DISCLAIMER = "Bản demo trên dữ liệu minh hoạ hư cấu, chưa phải ViHippoRAG đầy đủ."

# Bảng màu (giá trị lấy từ token Console – bản sáng; dùng cho canvas đồ thị)
COLOR = {
    "cụm từ": "#0e9087",      # data-2 teal
    "đoạn văn": "#085287",    # data-5 deep blue
    "văn bản": "#8d591e",     # data-3 amber
    "seed": "#5e6ad2",        # accent: điểm xuất phát
    "edge": "#c9c9d1",
    "synonym": "#9a9aa6",
    "sua_doi": "#85194c",     # data-4 wine
    "dan_chieu": "#878329",   # data-6 olive
    "can_cu": "#6b6b76",
    "ban_hanh": "#6b6b76",
    "thuoc_van_ban": "#d8c6ae",
    "text": "#1b1b20",
}
EDGE_LABEL = {"sua_doi": "sửa đổi", "dan_chieu": "dẫn chiếu", "can_cu": "căn cứ", "ban_hanh": "ban hành kèm theo",
              "thuoc_van_ban": "thuộc văn bản", "synonym": "đồng nghĩa", "fact": "bộ ba", "passage": "đoạn chứa cụm từ",
              "fact+synonym": "bộ ba"}

st.set_page_config(page_title="ViHippo – demo", layout="wide", initial_sidebar_state="auto")
st.markdown("""
<style>
  .block-container {padding-top: 3.5rem; max-width: 1280px;}
  h1, h2, h3 {font-weight: 600 !important; letter-spacing: -0.01em;}
  .lead {font-size: 1.08rem; line-height: 1.6; color: #3c3c46; max-width: 62rem; margin-bottom: 1rem;}
  .muted {color: #5d5d68; font-size: 0.9rem;}
  .tag {display: inline-block; padding: 1px 8px; border-radius: 5px; font-size: 0.8rem; font-weight: 500;
        background: rgba(0,0,0,0.05); color: #3c3c46; margin-right: 4px;}
  .tag-warn {background: rgba(166,122,12,0.12); color: #7d5c06;}
  .tag-acc {background: rgba(94,106,210,0.10); color: #4f5bc4;}
  .passage {padding: 10px 12px; border-radius: 8px; background: #ffffff; margin-bottom: 8px;
            box-shadow: 0 0 0 1px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);}
  .passage .cite {font-size: 0.82rem; color: #4f5bc4; font-weight: 500;}
  .passage .score {float: right; font-size: 0.78rem; color: #5d5d68; font-variant-numeric: tabular-nums;}
  .passage.old {background: #fbf8ef;}
  .answer {padding: 14px 16px; border-radius: 10px; background: #ffffff; line-height: 1.6;
           box-shadow: 0 0 0 1px rgba(0,0,0,0.07), 0 2px 6px rgba(0,0,0,0.05);}
  .legend span {display: inline-flex; align-items: center; margin-right: 16px; font-size: 0.85rem; color: #3c3c46;}
  .legend i {display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px;}
  .legend b {display: inline-block; width: 18px; height: 0; border-top: 2px solid; margin-right: 6px; font-weight: 400;}
  .stepnum {color: #4f5bc4; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ dữ liệu
@st.cache_data
def load_json(name: str):
    p = C.CACHE_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


corpus = load_json("corpus.json")
index_view = load_json("index_view.json")
samples = load_json("samples.json")
if not (corpus and index_view and samples):
    st.error("Chưa có dữ liệu lập chỉ mục trong thư mục cache/. Chạy `make index` (hoặc `python build_index.py`) "
             "khi có mạng, rồi mở lại trang này.")
    st.stop()

DOCS = corpus["docs"]
PASSAGES = corpus["passages"]
RELS = corpus["relations"]


@st.cache_resource(show_spinner="Đang nạp mô hình bge-m3 và đồ thị… (chỉ lần đầu)")
def get_engine():
    from vihippo.corpus import load_corpus
    from vihippo.engine import ViHippoRAG
    c = load_corpus(C.CORPUS_DIR, core_files=C.CORE_FILES, reference_date=C.REFERENCE_DATE)
    eng = ViHippoRAG(c)
    eng.prepare_retrieval_objects()
    return eng


def esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt_date(iso: str | None) -> str:
    if not iso:
        return "—"
    y, m, d = iso.split("-")
    return f"{d}/{m}/{y}"


def page_header(num: int | None, title: str, lead: str):
    prefix = f"<span class='stepnum'>Bước {num}.</span> " if num else ""
    st.markdown(f"## {prefix}{title}", unsafe_allow_html=True)
    st.markdown(f"<div class='lead'>{lead}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ đồ thị
def render_graph(nodes, edges, height=560, highlight_passages=(), sizes=None, physics=True):
    from pyvis.network import Network
    net = Network(height=f"{height}px", width="100%", bgcolor="#fbfbfc", font_color=COLOR["text"],
                  cdn_resources="in_line", directed=False)
    for n in nodes:
        t = n["type"]
        is_seed = n.get("seed", 0) > 0
        base = COLOR[t]
        size = (sizes or {}).get(n["id"], {"cụm từ": 10, "đoạn văn": 14, "văn bản": 22}[t])
        label = n["label"] if t != "đoạn văn" else n["label"].replace("Quy chế đào tạo trình độ ", "QC ").split(" (kèm")[0] + (
            ", " + ", ".join(n["label"].split(", ")[1:]) if "(kèm" in n["label"] else "")
        if t == "cụm từ" and len(label) > 34:
            label = label[:32] + "…"
        border = COLOR["seed"] if is_seed else ("#1b1b20" if n["id"] in highlight_passages else base)
        net.add_node(n["id"], label=label, title=f"{t}: {n['label']}", shape="dot" if t != "văn bản" else "square",
                     size=size, color={"background": COLOR["seed"] if is_seed else base, "border": border,
                                       "highlight": {"background": base, "border": "#1b1b20"}},
                     borderWidth=4 if (is_seed or n["id"] in highlight_passages) else 1,
                     font={"size": 13 if t != "cụm từ" else 11, "face": "Inter, system-ui, sans-serif"})
    ids = {n["id"] for n in nodes}
    for e in edges:
        if e["s"] not in ids or e["t"] not in ids:
            continue
        kind = e["kind"]
        color = COLOR.get(kind, COLOR["edge"])
        width = {"sua_doi": 4, "dan_chieu": 3, "can_cu": 2, "ban_hanh": 2}.get(kind, 1)
        net.add_edge(e["s"], e["t"], color=color, width=width, dashes=kind in ("synonym", "thuoc_van_ban"),
                     title=f"{EDGE_LABEL.get(kind, kind)} (trọng số {e['w']:.2f})")
    net.set_options(json.dumps({
        "physics": {"enabled": physics, "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {"gravitationalConstant": -45, "springLength": 90, "avoidOverlap": 0.4},
                    "stabilization": {"iterations": 250}},
        "interaction": {"hover": True, "tooltipDelay": 120, "navigationButtons": False},
        "edges": {"smooth": False},
    }))
    components.html(net.generate_html(notebook=False), height=height + 12, scrolling=False)


def legend(extra_seed=False, relations=True):
    items = [f"<span><i style='background:{COLOR['cụm từ']}'></i>cụm từ</span>",
             f"<span><i style='background:{COLOR['đoạn văn']}'></i>đoạn văn (Điều/Khoản)</span>",
             f"<span><i style='background:{COLOR['văn bản']};border-radius:2px'></i>văn bản</span>"]
    if extra_seed:
        items.append(f"<span><i style='background:{COLOR['seed']}'></i>nút hạt giống</span>")
    items.append(f"<span><b style='border-top:2px dashed {COLOR['synonym']}'></b>đồng nghĩa</span>")
    if relations:
        items += [f"<span><b style='border-color:{COLOR['sua_doi']};border-top-width:4px'></b>sửa đổi</span>",
                  f"<span><b style='border-color:{COLOR['dan_chieu']};border-top-width:3px'></b>dẫn chiếu</span>",
                  f"<span><b style='border-color:{COLOR['can_cu']}'></b>căn cứ / ban hành</span>"]
    st.markdown("<div class='legend'>" + "".join(items) + "</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ chọn câu hỏi
SAMPLE_BY_ID = {s["meta"]["id"]: s for s in samples.values()}
FREE = "__free__"


def question_picker(key: str):
    opts = [s["meta"]["id"] for s in samples.values()] + [FREE]
    labels = {s["meta"]["id"]: f"{s['meta']['label']}: {s['meta']['question']}" for s in samples.values()}
    labels[FREE] = "Câu hỏi tự do" + (" (cần mạng)" if OFFLINE else "")
    cur = st.session_state.get("qid", opts[0])
    qid = st.radio("Chọn câu hỏi", opts, index=opts.index(cur) if cur in opts else 0,
                   format_func=lambda x: labels[x], key=f"pick_{key}")
    st.session_state["qid"] = qid
    if qid != FREE:
        return SAMPLE_BY_ID[qid]
    if OFFLINE:
        st.info("Đang ở chế độ ngoại tuyến (--offline): hỏi tự do cần gọi LLM nên đã tắt. "
                "Ba câu mẫu vẫn chạy đầy đủ từ dữ liệu đã lưu.")
        return None
    q = st.text_input("Câu hỏi của bạn", value=st.session_state.get("free_q", ""),
                      placeholder="Ví dụ: Thời gian đào tạo tối đa của chương trình thạc sĩ là bao lâu?", key=f"free_{key}")
    st.session_state["free_q"] = q
    if not q.strip():
        st.caption("Nhập câu hỏi rồi nhấn Enter.")
        return None
    cache = st.session_state.setdefault("free_results", {})
    if q not in cache:
        with st.spinner("Đang truy xuất trên đồ thị…"):
            cache[q] = run_question(get_engine(), q)
            cache[q]["meta"] = {"id": FREE, "label": "Tự do", "question": q, "expected": None, "why": None}
    return cache[q]


def options_panel():
    with st.sidebar.expander("Tuỳ chọn thử nghiệm", expanded=False):
        st.caption(f"Mặc định: top-{C.TOP_K} (như HippoRAG 2), chỉ giảm điểm đoạn đã bị sửa, thiếu căn cứ thì từ chối.")
        k = st.radio("Số đoạn đưa cho LLM", [3, 5], index=[3, 5].index(C.TOP_K), horizontal=True, key="opt_k")
        inherit = st.toggle("Đoạn sửa đổi kế thừa điểm của đoạn bị sửa", value=False, key="opt_inh",
                            help="Quy tắc hiệu lực mở rộng (không có trong đặc tả gốc): nếu một khoản cũ được tìm thấy, "
                                 "khoản sửa đổi nó nhận ít nhất bằng điểm gốc của khoản cũ.")
        partial = st.toggle("Cho phép trả lời một phần", value=False, key="opt_partial",
                            help="Khi chỉ đủ căn cứ một phần, LLM trả lời phần đó và nói rõ phần nào văn bản không quy định.")
    return k, inherit, ("partial" if partial else "strict")


def fetch_answer(res, side, k, inherit, mode):
    ans = get_answer(None, res, side, k, inherit, mode)
    if ans is None and not OFFLINE:
        with st.spinner("LLM đang trả lời…"):
            ans = get_answer(get_engine(), res, side, k, inherit, mode)
    return ans


def passage_html(p, show_note=True, rank=None, score_label=None):
    old = show_note and p.get("note")
    note = f"<div style='margin-top:6px'><span class='tag tag-warn'>Đã bị sửa đổi</span>" \
           f"<span class='muted'>{esc(p['note'])}</span></div>" if old else ""
    sc = f"<span class='score'>{score_label}</span>" if score_label else ""
    r = f"{rank}. " if rank else ""
    return (f"<div class='passage{' old' if old else ''}'>{sc}<div class='cite'>{r}[{esc(p['citation'])}]</div>"
            f"<div>{esc(p['text'])}</div>{note}</div>")


def answer_html(text: str) -> str:
    import re
    t = esc(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(\[[^\]]+\])", r"<span style='color:#4f5bc4'>\1</span>", t)
    return f"<div class='answer'>{t.replace(chr(10), '<br>')}</div>"


# ------------------------------------------------------------------ trang
def page_why():
    page_header(1, "Vì sao cần RAG",
                "Mô hình ngôn ngữ (LLM) giống một người rất giỏi tiếng Việt nhưng <b>chưa từng đọc quy chế của trường</b>. "
                "Hỏi thẳng thì nó đoán. RAG là cách <b>đưa đúng vài trang tài liệu cho nó đọc trước</b> rồi mới trả lời, "
                "kèm chỉ rõ trang nào.")
    s = SAMPLE_BY_ID["q2_amended"]
    q = s["meta"]["question"]
    st.markdown(f"**Thử hỏi:** “{q}”")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### LLM chưa đọc tài liệu")
        st.markdown(answer_html(s["no_rag"]["text"]), unsafe_allow_html=True)
        st.caption("Không có tài liệu, LLM chỉ có thể đoán hoặc từ chối: nó không biết quy chế của trường nào, văn bản nào còn hiệu lực.")
    with c2:
        st.markdown("#### LLM được đọc tài liệu trước (ViHippo)")
        st.markdown(answer_html(s["answers"][f"vihippo|k{C.TOP_K}|inh0|strict"]["text"]), unsafe_allow_html=True)
        st.caption("Mỗi ý đều có trích dẫn [văn bản, Điều, Khoản] để người dùng tự kiểm tra lại.")
    st.markdown("#### RAG làm ba việc")
    a, b, c = st.columns(3)
    a.markdown("**1. Tìm**  \nTìm vài đoạn văn bản liên quan nhất tới câu hỏi trong kho.")
    b.markdown("**2. Đọc**  \nĐưa các đoạn đó cho LLM cùng câu hỏi.")
    c.markdown("**3. Trả lời có trích dẫn**  \nLLM chỉ được dùng các đoạn đó; thiếu căn cứ thì từ chối.")
    st.markdown("#### Vậy vấn đề ở đâu?")
    st.markdown(
        "RAG thường tìm đoạn **giống câu hỏi nhất về mặt chữ nghĩa**. Với văn bản hành chính, câu trả lời đúng hay nằm ở "
        "**một văn bản khác**, viết bằng **từ ngữ khác**: một thông báo chỉ ghi “theo Quy chế hiện hành”, còn quy chế thì "
        "đã bị một quyết định sửa đổi. ViHippo dựa trên HippoRAG 2: dựng một **bản đồ quan hệ** (đồ thị tri thức) giữa các "
        "đoạn và văn bản, rồi **lần theo quan hệ** để tìm.")


def page_corpus():
    page_header(2, "Kho văn bản",
                "Kho gồm 5 văn bản có liên quan với nhau, 1 văn bản căn cứ bên ngoài (chỉ có số, ký hiệu, trích yếu) "
                "và 14 văn bản cùng trường để gây nhiễu. Tất cả là <b>hư cấu</b>, của “Trường Đại học ABC”.")
    core = [d for d in DOCS.values() if d["is_core"] and not d["external"]]
    ext = [d for d in DOCS.values() if d["external"] and d["so"] == "23/2021/TT-BGDĐT"]
    noise = [d for d in DOCS.values() if not d["is_core"] and not d["external"]]
    m = st.columns(4)
    m[0].metric("Văn bản trong kho", len([d for d in DOCS.values() if not d["external"]]))
    m[1].metric("Đoạn (Điều/Khoản)", len(PASSAGES))
    m[2].metric("Văn bản ngoài kho", len([d for d in DOCS.values() if d["external"]]))
    m[3].metric("Quan hệ rút bằng regex", len(RELS))

    def rel_lines(doc_id):
        out = []
        for r in RELS:
            if r["src_doc"] == doc_id:
                tgt = DOCS[r["dst_doc"]]["short"]
                where = ""
                if r.get("src_pid"):
                    where = PASSAGES[r["src_pid"]]["citation"].split(", ", 1)[-1] + ": "
                if r.get("dst_pid"):
                    tgt = PASSAGES[r["dst_pid"]]["citation"]
                out.append(f"<span class='tag'>{EDGE_LABEL[r['kind']]}</span> {esc(where)}→ {esc(tgt)}")
        return out

    cards = core + ext
    for row in range(0, len(cards), 3):
        cols = st.columns(3, gap="medium")
        for col, d in zip(cols, cards[row:row + 3]):
            with col.container(border=True):
                kind = "Văn bản căn cứ bên ngoài" if d["external"] else d["loai"].capitalize()
                st.markdown(f"<span class='tag'>{esc(kind)}</span>", unsafe_allow_html=True)
                st.markdown(f"**{esc(d['short'])}**")
                st.markdown(f"<div class='muted'>{esc(d['trich_yeu'])}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='muted' style='margin-top:6px'>Ban hành {fmt_date(d['ngay_ban_hanh'])}"
                            + (f" · Hiệu lực {fmt_date(d['hieu_luc'])}" if not d["external"] else " · Chỉ là nút, không lập chỉ mục")
                            + (f" · {len(d['passages'])} đoạn" if d["passages"] else "") + "</div>", unsafe_allow_html=True)
                rl = rel_lines(d["doc_id"])
                if rl:
                    st.markdown("<div style='margin-top:8px;font-size:0.88rem;line-height:1.7'>" + "<br>".join(rl) + "</div>",
                                unsafe_allow_html=True)
                if d.get("raw"):
                    with st.expander("Xem toàn văn"):
                        st.text(d["raw"])
    st.markdown("#### Quan hệ giữa các văn bản")
    st.caption("Rút bằng biểu thức chính quy từ thể thức văn bản: dòng “Căn cứ…”, “Ban hành kèm theo…”, "
               "“Sửa đổi khoản … Điều …”, “theo … hiện hành”. Không dùng LLM ở bước này.")
    nodes, edges = [], []
    ids = {}
    for i, d in enumerate([d for d in DOCS.values() if d["is_core"] or d["external"] or any(
            r["src_doc"] == d["doc_id"] or r["dst_doc"] == d["doc_id"] for r in RELS)]):
        ids[d["doc_id"]] = i
        nodes.append({"id": i, "type": "văn bản", "label": d["short"]})
    for r in RELS:
        if r["src_doc"] in ids and r["dst_doc"] in ids:
            edges.append({"s": ids[r["src_doc"]], "t": ids[r["dst_doc"]], "w": r["weight"], "kind": r["kind"]})
    render_graph(nodes, edges, height=380)
    legend(relations=True)
    with st.expander(f"{len(noise)} văn bản gây nhiễu (cùng trường, cùng chủ đề đào tạo, học phí, tuyển sinh…)"):
        st.dataframe(pd.DataFrame([{"Văn bản": d["short"], "Trích yếu": d["trich_yeu"], "Số đoạn": len(d["passages"]),
                                    "Ban hành": fmt_date(d["ngay_ban_hanh"])} for d in noise]),
                     hide_index=True, use_container_width=True)


def page_index():
    page_header(3, "Lập chỉ mục: từ văn bản thành bản đồ quan hệ",
                "Máy không “đọc hiểu” cả văn bản một lúc. Ta cắt văn bản thành từng Điều/Khoản, nhờ LLM ghi lại các "
                "<b>sự kiện dạng (ai/cái gì – làm gì – với cái gì)</b>, rồi nối tất cả thành một bản đồ. "
                "Bước này chạy một lần và được lưu lại, lúc demo không gọi LLM nữa.")
    stats = index_view["stats"]
    m = st.columns(5)
    m[0].metric("Đoạn văn", stats["passages"])
    m[1].metric("Cụm từ (nút)", stats["phrases"])
    m[2].metric("Bộ ba", stats["facts"])
    m[3].metric("Cạnh đồng nghĩa", stats["synonym_edges"])
    m[4].metric("Tổng số cạnh", stats["edges"])

    st.markdown("#### 3a. Cắt theo Điều/Khoản (regex)")
    core_docs = [d for d in DOCS.values() if d["is_core"] and not d["external"]]
    pick = st.selectbox("Văn bản", [d["doc_id"] for d in core_docs], format_func=lambda x: DOCS[x]["short"],
                        index=1, key="idx_doc")
    doc = DOCS[pick]
    df = pd.DataFrame([{"Trích dẫn": PASSAGES[p]["citation"].replace(doc["short"] + ", ", ""),
                        "Nội dung": PASSAGES[p]["text"],
                        "Hiệu lực": ("Đã bị sửa: " + PASSAGES[p]["superseded_note"].replace("đã bị sửa bởi ", ""))
                        if PASSAGES[p]["superseded_note"] else "—"} for p in doc["passages"]])
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown("#### 3b. Mỗi đoạn → các bộ ba (chủ thể, quan hệ, đối tượng)")
    st.caption("Do LLM rút ra bằng prompt OpenIE tiếng Việt (thay prompt tiếng Anh của HippoRAG 2).")
    pid = st.selectbox("Đoạn", doc["passages"], format_func=lambda x: PASSAGES[x]["citation"], key="idx_pid",
                       index=min(len(doc["passages"]) - 1, next((i for i, p in enumerate(doc["passages"]) if PASSAGES[p]["superseded_note"]), 0)))
    st.markdown(passage_html({**PASSAGES[pid], "note": PASSAGES[pid]["superseded_note"]}), unsafe_allow_html=True)
    tr = index_view["triples"].get(pid, {"triples": [], "entities": []})
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.markdown("**Thực thể nhận diện được**")
        st.markdown(" ".join(f"<span class='tag'>{esc(e)}</span>" for e in tr["entities"]) or "—", unsafe_allow_html=True)
    with c2:
        st.markdown("**Bộ ba**")
        st.dataframe(pd.DataFrame(tr["triples"], columns=["Chủ thể", "Quan hệ", "Đối tượng"]),
                     hide_index=True, use_container_width=True)

    st.markdown("#### 3c. Ghép thành đồ thị")
    st.markdown("Mỗi **cụm từ** thành một nút; bộ ba nối hai cụm từ; mỗi **đoạn** nối với các cụm từ nó chứa; "
                "các cụm từ gần nghĩa (theo bge-m3) được nối bằng **cạnh đồng nghĩa**. ViHippo thêm **nút văn bản** "
                "và các **cạnh quan hệ có trọng số**: sửa đổi (3) > dẫn chiếu (2) > căn cứ (1).")
    g = index_view["core_graph"]
    full = st.toggle("Hiện toàn bộ phần đồ thị của 5 văn bản lõi", value=False, key="idx_full")
    if not full:
        focus = {pid for pid, p in PASSAGES.items() if p["superseded_by"]}
        focus |= {p["superseded_by"] for p in PASSAGES.values() if p["superseded_by"]}
        focus |= {pid for pid in PASSAGES if pid.startswith("58/TB-ĐHABC#M3")}
        pv = {n["id"] for n in g["nodes"] if n["type"] == "đoạn văn" and n.get("pid") in focus}
        ent = set()
        for e in g["edges"]:
            for a, b in ((e["s"], e["t"]), (e["t"], e["s"])):
                if a in pv:
                    ent.add(b)
        keep_docs = {n["id"] for n in g["nodes"] if n["type"] == "văn bản" and n["label"].split(" (kèm")[0] in
                     {"QĐ 100/QĐ-ĐHABC", "QĐ 245/QĐ-ĐHABC", "TB 58/TB-ĐHABC", "Quy chế đào tạo trình độ thạc sĩ"}}
        keep = pv | ent | keep_docs
        nodes = [n for n in g["nodes"] if n["id"] in keep and (n["type"] != "đoạn văn" or n["id"] in pv)]
        render_graph(nodes, g["edges"], height=600)
        st.caption("Đang thu gọn: chỉ vẽ 2 khoản bị sửa, 2 khoản sửa đổi, mục dẫn chiếu của TB 58 và các nút nối trực tiếp với chúng.")
    else:
        render_graph(g["nodes"], g["edges"], height=680)
    legend(relations=True)
    st.caption("Rê chuột lên nút/cạnh để xem chi tiết; kéo thả được.")


def compare_columns(res, k, inherit, mode):
    meta = res["meta"]
    c1, c2 = st.columns(2, gap="large")
    for col, side, title, sub in [
        (c1, "baseline", "RAG thường", f"Chỉ so khớp vector (bge-m3), lấy {k} đoạn giống câu hỏi nhất."),
        (c2, "vihippo", "ViHippo", f"HippoRAG 2 + quan hệ văn bản + hiệu lực, lấy {k} đoạn có điểm PPR cao nhất."),
    ]:
        with col:
            st.markdown(f"### {title}")
            st.caption(sub)
            ps = passages_for(res, side, k, inherit)
            st.markdown("".join(passage_html(p, show_note=(side == "vihippo"), rank=i + 1)
                                for i, p in enumerate(ps)), unsafe_allow_html=True)
            ans = fetch_answer(res, side, k, inherit, mode)
            st.markdown("**Câu trả lời**")
            if ans:
                st.markdown(answer_html(ans["text"]), unsafe_allow_html=True)
            else:
                st.info("Biến thể này chưa có trong dữ liệu đã lưu (chế độ ngoại tuyến).")
    if meta.get("expected"):
        with st.expander("Đáp án mong đợi và vì sao câu này khó"):
            st.markdown(f"**Đáp án:** {meta['expected']}  \n**Vì sao khó:** {meta['why']}")
            ab = res.get("ablation") or []
            van = next((r for r in ab if r.get("answer")), None)
            if van:
                st.markdown(f"**HippoRAG 2 gốc (top-{C.TOP_K}, không lớp văn bản, không nhãn hiệu lực) trả lời:**")
                st.markdown(answer_html(van["answer"]["text"]), unsafe_allow_html=True)
            if ab:
                st.markdown("**Thử gỡ dần từng thành phần**: hạng của các đoạn cần tìm trong kết quả truy xuất")
                st.dataframe(pd.DataFrame([{"Cấu hình": r["name"], **{PASSAGES[g]["citation"]: v for g, v in r["gold_ranks"].items()}}
                                           for r in ab]), hide_index=True, use_container_width=True)


def why_ppr_box(res):
    """Khung giải thích cho câu đa bước: số liệu lấy từ cache, không viết tay."""
    v = res["vihippo"]
    ab = res.get("ablation") or []
    with st.container(border=True):
        st.markdown("#### Vì sao PPR chạy sai ở câu này")
        st.markdown("<span class='tag tag-warn'>Ý tưởng cải tiến, chưa phải kết quả</span> "
                    "<span class='muted'>Phân tích trên 3 câu mẫu; chưa đánh giá trên bộ câu hỏi nào.</span>",
                    unsafe_allow_html=True)

        st.markdown("**Nguyên nhân 1: cụm từ ngày tháng hiếm kéo điểm.** HippoRAG 2 chia điểm của mỗi bộ ba cho số đoạn "
                    "chứa cụm từ, nên cụm từ chỉ xuất hiện ở một đoạn (như một ngày cụ thể) giữ trọn điểm và trở thành hạt "
                    "giống nặng nhất. Mực dồn vào TB 58 Mục 2 (ngày bảo vệ), không phải vào chuỗi quy chế → quyết định sửa đổi.")
        phr = [x for x in v.get("seeds", []) if x["type"] == "cụm từ"]
        if phr:
            st.dataframe(pd.DataFrame([{"Trọng số xuất phát": round(x["weight"], 3), "Cụm từ hạt giống": x["label"]} for x in phr]),
                         hide_index=True, use_container_width=True)
        has_ref = any("hiện hành" in " ".join(c["fact"]) for c in v["candidates"])
        if not has_ref:
            st.caption("Bộ ba “nộp hồ sơ theo… Quy chế hiện hành” (TB 58 Mục 3) không nằm trong 10 bộ ba khớp nhất, "
                       "nên mắt xích dẫn chiếu không được dùng làm hạt giống.")

        st.markdown("**Nguyên nhân 2: cạnh sửa đổi bị chia loãng.** Ở mỗi vòng, điểm của một nút chia cho mọi cạnh của nó "
                    "theo trọng số. Cạnh “sửa đổi” (trọng số 3) chỉ là một phần nhỏ trong tổng trọng số cạnh của khoản cũ; "
                    "phần còn lại chảy sang cụm từ và sang nút văn bản, rồi từ nút văn bản toả ra mọi khoản khác của Quy chế.")
        share = res.get("amend_share") or []
        if share:
            st.dataframe(pd.DataFrame([{"Khoản cũ": r["old"].replace("Quy chế đào tạo trình độ thạc sĩ (kèm QĐ 100/QĐ-ĐHABC)", "Quy chế ThS"),
                                        "Số cạnh": r["n_edges"], "Tổng trọng số": round(r["total_weight"], 1),
                                        "Cạnh sửa đổi": round(r["amend_weight"], 1), "Tỷ lệ": f"{r['share'] * 100:.0f}%"} for r in share]),
                         hide_index=True, use_container_width=True)
        if ab:
            st.markdown("Hạng của các đoạn cần tìm khi gỡ dần từng thành phần:")
            st.dataframe(pd.DataFrame([{"Cấu hình": r["name"], **{PASSAGES[g]["citation"].replace(
                "Quy chế đào tạo trình độ thạc sĩ (kèm QĐ 100/QĐ-ĐHABC)", "Quy chế ThS"): x for g, x in r["gold_ranks"].items()}}
                for r in ab]), hide_index=True, use_container_width=True)

        st.markdown("**Thử nghiệm: bật “kế thừa điểm”.** Đoạn sửa đổi nhận ít nhất điểm gốc của khoản mà nó sửa. "
                    "Điểm PPR của các nút không đổi; quy tắc này chỉ xếp lại các đoạn sau PPR.")
        before = v.get("passages_top10") or v["passages"]
        after = v.get("passages_inherit_top10") or v.get("passages_inherit") or []
        short = lambda c: c.replace("Quy chế đào tạo trình độ thạc sĩ (kèm QĐ 100/QĐ-ĐHABC)", "Quy chế ThS").replace(
            "Quy chế đào tạo trình độ tiến sĩ (kèm QĐ 101/QĐ-ĐHABC)", "Quy chế TS")
        rows = []
        for i in range(min(6, len(before), len(after))):
            rows.append({"Hạng": i + 1, "Trước (mặc định)": short(before[i]["citation"]), "Điểm trước": round(before[i]["score"], 4),
                         "Sau (kế thừa điểm)": short(after[i]["citation"]), "Điểm sau": round(after[i]["score"], 4)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        a_off = (res.get("answers") or {}).get(f"vihippo|k{C.TOP_K}|inh0|strict")
        a_on = (res.get("answers") or {}).get(f"vihippo|k{C.TOP_K}|inh1|strict")
        if a_off and a_on:
            st.markdown(f"Câu trả lời của ViHippo (top-{C.TOP_K}, trả lời chặt) **trước** và **sau** khi bật kế thừa điểm:")
            st.markdown(answer_html(a_off["text"]), unsafe_allow_html=True)
            st.markdown(answer_html(a_on["text"]), unsafe_allow_html=True)


def page_ask(k, inherit, mode):
    page_header(4, "Hỏi thử và so sánh",
                "Cùng một câu hỏi, cùng mô hình embedding, cùng LLM và cùng cách hỏi LLM. Chỉ khác <b>cách tìm tài liệu</b>: "
                "bên trái tìm theo độ giống chữ nghĩa, bên phải lần theo bản đồ quan hệ.")
    res = question_picker("ask")
    if res is None:
        return
    compare_columns(res, k, inherit, mode)
    if res["meta"].get("id") == "q1_multihop":
        why_ppr_box(res)


def page_inside(k, inherit, mode):
    page_header(5, "Bên trong ViHippo: các bước truy xuất",
                "Đây là những gì xảy ra trong vài giây sau khi bấm hỏi. Ý tưởng giống thả một giọt mực vào đúng vài điểm "
                "trên bản đồ rồi để mực <b>loang theo các đường nối</b>: đoạn nào thấm nhiều mực nhất thì được chọn.")
    res = question_picker("inside")
    if res is None:
        return
    v = res["vihippo"]
    st.markdown("#### (a) Câu hỏi khớp với những bộ ba nào")
    st.caption("So câu hỏi với toàn bộ bộ ba bằng embedding. 5 bộ ba đầu được đưa sang bước (b).")
    st.dataframe(pd.DataFrame([{"Độ khớp": round(c["score"], 3), "Đưa cho LLM lọc": "có" if c["to_llm"] else "—",
                                "Chủ thể": c["fact"][0], "Quan hệ": c["fact"][1], "Đối tượng": c["fact"][2]}
                               for c in v["candidates"]]), hide_index=True, use_container_width=True,
                 column_config={"Độ khớp": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})

    st.markdown("#### (b) LLM lọc, giữ lại bộ ba nào và vì sao")
    st.caption("Bước “nhận diện” của HippoRAG 2 (thay cho nhận diện thực thể trong câu hỏi), dùng prompt tiếng Việt.")
    if v["kept"]:
        st.dataframe(pd.DataFrame(v["kept"], columns=["Chủ thể", "Quan hệ", "Đối tượng"]), hide_index=True,
                     use_container_width=True)
    else:
        st.warning("LLM không giữ bộ ba nào. ViHippo quay về tìm theo vector (giống HippoRAG 2 gốc).")
    if v.get("filter_reasons"):
        st.markdown("**Lý do LLM đưa ra**")
        st.markdown(f"<div class='muted' style='white-space:pre-wrap'>{esc(v['filter_reasons'])}</div>", unsafe_allow_html=True)

    if not v.get("seeds"):
        st.info(v.get("fallback", ""))
    else:
        st.markdown("#### (c) Nút hạt giống: nơi “nhỏ mực”")
        st.caption("Cụm từ trong các bộ ba được giữ thành điểm xuất phát; cụm từ càng hiếm (ít đoạn chứa) càng nặng ký. "
                   f"Ngoài ra mọi đoạn văn đều nhận một lượng nhỏ theo độ giống câu hỏi (×0,05) — có {v['n_passage_seeds']} đoạn như vậy, bảng chỉ hiện 5 đoạn đầu.")
        st.dataframe(pd.DataFrame([{"Trọng số xuất phát": round(s["weight"], 4), "Loại nút": s["type"], "Nút": s["label"]}
                                   for s in v["seeds"]]), hide_index=True, use_container_width=True)

        st.markdown("#### (d) Điểm PPR lan trên đồ thị")
        it = v.get("iterations")
        step = 0
        if it:
            last = len(it["scores"]) - 1
            step = st.slider("Vòng lặp lan truyền", 0, last, last, key=f"ppr_step_{res['meta']['id']}",
                             help="Vòng 0: chỉ có hạt giống. Mỗi vòng, một nửa điểm của mỗi nút chia cho hàng xóm theo trọng số cạnh, "
                                  "nửa còn lại quay về hạt giống (damping = 0,5).")
            scores = it["scores"][step]
            top = sorted(zip(it["nodes"], scores, it["node_idx"]), key=lambda x: -x[1])[:15]
            fig = go.Figure(go.Bar(
                x=[s for _, s, _ in top][::-1], y=[f"{n['label'][:48]}" for n, _, _ in top][::-1], orientation="h",
                marker_color=[COLOR[n["type"]] for n, _, _ in top][::-1],
                hovertemplate="%{y}<br>điểm %{x:.4f}<extra></extra>"))
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#fbfbfc", paper_bgcolor="#fbfbfc",
                              xaxis=dict(title="điểm PPR", gridcolor="#e6e6ea"), font=dict(family="Inter, system-ui", size=12))
            c1, c2 = st.columns([1, 1], gap="large")
            with c1:
                st.markdown(f"**Top nút ở vòng {step}**")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                legend(extra_seed=False, relations=False)
            with c2:
                st.markdown("**Bảng top nút (kết quả PPR cuối)**")
                st.dataframe(pd.DataFrame([{"Điểm PPR": round(n["ppr"], 4), "Loại": n["type"], "Nút": n["label"][:60],
                                            "Hạt giống": "có" if n["is_seed"] else "—"} for n in v["top_nodes"][:15]]),
                             hide_index=True, use_container_width=True, height=430,
                             column_config={"Điểm PPR": st.column_config.NumberColumn(format="%.4f", width="small"),
                                            "Loại": st.column_config.TextColumn(width="small")})
            g = v["graph"]
            idx_score = dict(zip(it["node_idx"], scores))
            mx = max(scores) or 1
            sizes = {n["id"]: 8 + 34 * (idx_score.get(n["id"], 0) / mx) ** 0.6 for n in g["nodes"]}
            st.markdown("**Đồ thị con quanh hạt giống** (kích thước nút = điểm ở vòng đang chọn; viền đen = đoạn được chọn)")
            render_graph(g["nodes"], g["edges"], height=560, highlight_passages=set(g.get("top_passages", [])), sizes=sizes)
            legend(extra_seed=True, relations=True)

    st.markdown(f"#### (e) Top {k} đoạn văn")
    st.caption("Điểm cuối = điểm PPR; đoạn đã bị sửa đổi vẫn giữ lại nhưng điểm × "
               f"{C.SUPERSEDED_FACTOR:g} và có nhãn" + ("; đang bật quy tắc kế thừa điểm." if inherit else "."))
    ps = passages_for(res, "vihippo", k, inherit)
    st.markdown("".join(passage_html(p, rank=i + 1, score_label=(
        f"điểm {p['score']:.4f}" + (f" (gốc {p['raw_score']:.4f})" if p.get("raw_score") and abs(p["raw_score"] - p["score"]) > 1e-9 else "")))
        for i, p in enumerate(ps)), unsafe_allow_html=True)
    ans = fetch_answer(res, "vihippo", k, inherit, mode)
    if ans:
        st.markdown("**Câu trả lời có trích dẫn**")
        st.markdown(answer_html(ans["text"]), unsafe_allow_html=True)
    if res["meta"].get("id") == "q1_multihop":
        why_ppr_box(res)


GLOSSARY = [
    ("RAG (sinh văn bản tăng cường truy xuất)",
     "Cho mô hình ngôn ngữ đọc đúng vài đoạn tài liệu trước khi trả lời, thay vì để nó trả lời theo trí nhớ.",
     "Như cho phép thí sinh mở đúng trang quy chế cần thiết khi làm bài, thay vì bắt học thuộc cả cuốn."),
    ("Embedding",
     "Biến một câu thành một dãy số sao cho các câu cùng nghĩa có dãy số gần nhau.",
     "“Hội đồng chấm có 03 người” và “hội đồng đánh giá gồm 3 thành viên” cho ra hai dãy số gần nhau dù chữ khác nhau."),
    ("Đồ thị tri thức",
     "Một bản đồ trong đó mỗi điểm là một khái niệm hay một đoạn văn bản, mỗi đường nối là một quan hệ giữa chúng.",
     "Điểm “QĐ 245” nối với điểm “Quy chế thạc sĩ” bằng đường “sửa đổi”."),
    ("Bộ ba (chủ thể, quan hệ, đối tượng)",
     "Một sự kiện nhỏ được viết gọn thành ba phần, để máy dễ ghép thành bản đồ.",
     "(hội đồng chấm, có, 03 người) — rút ra từ Khoản 2 Điều 1 của QĐ 245."),
    ("PPR (Personalized PageRank)",
     "Thả “mực” vào vài điểm xuất phát liên quan tới câu hỏi rồi cho mực loang theo đường nối; nơi thấm nhiều mực nhất là nơi quan trọng nhất với câu hỏi đó.",
     "Mực nhỏ vào “05 thành viên” loang sang khoản Điều 7, rồi theo đường “sửa đổi” sang QĐ 245."),
]


def page_glossary():
    page_header(None, "Thuật ngữ", "Năm khái niệm cần nắm để theo dõi bản demo, mỗi khái niệm một câu và một ví dụ.")
    for term, meaning, example in GLOSSARY:
        st.markdown(f"**{term}**  \n{meaning}  \n<span class='muted'>Ví dụ: {esc(example)}</span>", unsafe_allow_html=True)
        st.markdown("")


# ------------------------------------------------------------------ khung
st.sidebar.markdown("### ViHippo")
st.sidebar.caption("Truy xuất đa bước trên văn bản hành chính tiếng Việt, dựa trên HippoRAG 2")
PAGES = {
    "1. Vì sao cần RAG": page_why,
    "2. Kho văn bản": page_corpus,
    "3. Lập chỉ mục": page_index,
    "4. Hỏi thử và so sánh": page_ask,
    "5. Bên trong ViHippo": page_inside,
    "Thuật ngữ": page_glossary,
}
_names = list(PAGES)
if "page" not in st.session_state:  # trang hiện tại nằm trong URL (?p=1..6) để mở thẳng khi trình bày
    try:
        st.session_state["page"] = _names[int(st.query_params.get("p", "1")) - 1]
    except (ValueError, IndexError):
        st.session_state["page"] = _names[0]
page = st.sidebar.radio("Trình bày", _names, key="page", label_visibility="collapsed")
st.query_params["p"] = str(_names.index(page) + 1)
k, inherit, mode = options_panel()
st.sidebar.markdown("---")
st.sidebar.caption(("Chế độ ngoại tuyến: dùng dữ liệu đã lưu. " if OFFLINE else "")
                   + f"Ngày xét hiệu lực: {fmt_date(C.REFERENCE_DATE)}")
st.sidebar.warning(DISCLAIMER)
st.caption(DISCLAIMER)

fn = PAGES[page]
if fn in (page_ask, page_inside):
    fn(k, inherit, mode)
else:
    fn()
