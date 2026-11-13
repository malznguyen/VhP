"""Đọc kho văn bản hành chính, tách theo Điều/Khoản và rút quan hệ giữa văn bản bằng regex.

Không dùng LLM ở bước này: mọi thứ (số ký hiệu, ngày, hiệu lực, căn cứ, sửa đổi,
dẫn chiếu) đều lấy từ thể thức văn bản theo Nghị định 30/2020/NĐ-CP.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

DOC_TYPES = {
    "QUYẾT ĐỊNH": "QĐ",
    "QUY CHẾ": "Quy chế",
    "QUY ĐỊNH": "Quy định",
    "THÔNG BÁO": "TB",
    "HƯỚNG DẪN": "HD",
    "KẾ HOẠCH": "KH",
    "CÔNG VĂN": "CV",
    "THÔNG TƯ": "Thông tư",
    "NGHỊ QUYẾT": "Nghị quyết",
}

# Trọng số cạnh giữa văn bản (sửa đổi > dẫn chiếu > căn cứ), đưa vào đồ thị PPR.
RELATION_WEIGHTS = {
    "sua_doi": 3.0,
    "ban_hanh": 2.0,
    "dan_chieu": 2.0,
    "can_cu": 1.0,
}
RELATION_LABELS = {
    "sua_doi": "sửa đổi",
    "ban_hanh": "ban hành kèm theo",
    "dan_chieu": "dẫn chiếu",
    "can_cu": "căn cứ",
}

RE_SO = re.compile(r"^Số:\s*(\S+)", re.M)
RE_NGAY = re.compile(r"ngày (\d{1,2}) tháng (\d{1,2}) năm (\d{4})")
RE_KEM_THEO = re.compile(r"[Bb]an hành kèm theo Quyết định số (\S+?) ngày (\d{1,2})/(\d{1,2})/(\d{4})")
RE_CAN_CU = re.compile(r"^Căn cứ (.+?)[;.]?$", re.M)
RE_SO_VB = re.compile(r"(Thông tư|Nghị định|Nghị quyết|Quyết định|Hướng dẫn|Thông báo|Kế hoạch|Công văn) số\s+(\d+(?:/\d{4})?/[^\s,;.)]+)")
RE_HIEU_LUC = re.compile(r"có hiệu lực (?:thi hành )?(?:kể )?từ ngày (\d{1,2})/(\d{1,2})/(\d{4})")
RE_DIEU = re.compile(r"^Điều (\d+)\.\s*(.*)$")
RE_MUC = re.compile(r"^(\d+)\.\s+(.*)$")
RE_SUA_DOI = re.compile(r"[Ss]ửa đổi(?:, bổ sung)? khoản (\d+) Điều (\d+)")
RE_HIEN_HANH = re.compile(r"(Quy chế|Quy định|Hướng dẫn) ([^,.;:“”\"]+?) hiện hành")
RE_SUA_DOI_TARGET = re.compile(r"của (?:Quy chế|Quy định) .+? ban hành kèm theo Quyết định số (\S+)")


def norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold().strip()


@dataclass
class Passage:
    pid: str
    doc_id: str
    dieu: Optional[int]
    dieu_title: Optional[str]
    khoan: Optional[int]
    unit: str  # "Khoản" | "Điều" | "Mục"
    text: str
    citation: str = ""
    content: str = ""  # văn bản đưa vào HippoRAG (có tiêu đề ngữ cảnh)
    superseded_by: Optional[str] = None  # pid của đoạn sửa đổi
    superseded_note: Optional[str] = None


@dataclass
class Document:
    doc_id: str
    file: Optional[str]
    loai: str
    so: Optional[str]
    trich_yeu: str
    ngay_ban_hanh: Optional[str]
    hieu_luc: Optional[str]
    kem_theo: Optional[str] = None
    external: bool = False
    is_core: bool = False
    raw: str = ""
    passages: List[str] = field(default_factory=list)

    @property
    def short(self) -> str:
        abbr = DOC_TYPES.get(self.loai, self.loai)
        if self.so:
            return f"{abbr} {self.so}"
        if self.kem_theo:
            return f"{abbr} {self.trich_yeu.lower()} (kèm QĐ {self.kem_theo})"
        return f"{abbr} {self.trich_yeu}"


@dataclass
class Relation:
    kind: str
    src_doc: str
    dst_doc: str
    src_pid: Optional[str] = None
    dst_pid: Optional[str] = None
    evidence: str = ""

    @property
    def weight(self) -> float:
        return RELATION_WEIGHTS[self.kind]


@dataclass
class Corpus:
    docs: Dict[str, Document]
    passages: Dict[str, Passage]
    relations: List[Relation]
    reference_date: str

    def passage_list(self) -> List[Passage]:
        return list(self.passages.values())

    def by_content(self) -> Dict[str, Passage]:
        return {p.content: p for p in self.passages.values()}

    def to_json(self) -> dict:
        return {
            "reference_date": self.reference_date,
            "docs": {k: {**asdict(v), "short": v.short} for k, v in self.docs.items()},
            "passages": {k: asdict(v) for k, v in self.passages.items()},
            "relations": [{**asdict(r), "weight": r.weight} for r in self.relations],
        }


def _iso(d: str, m: str, y: str) -> str:
    return date(int(y), int(m), int(d)).isoformat()


def _parse_header(raw: str, lines: List[str], body_start: int):
    header = "\n".join(lines[:body_start])
    so = (m.group(1) if (m := RE_SO.search(header)) else None)
    ngay = (_iso(*m.groups()) if (m := RE_NGAY.search(header)) else None)
    loai, trich_yeu = "", ""
    for i, line in enumerate(lines[:body_start]):
        if line.strip() in DOC_TYPES:
            loai = line.strip()
            trich_yeu = lines[i + 1].strip() if i + 1 < len(lines) else ""
            break
    kem_theo = None
    if m := RE_KEM_THEO.search(header):
        kem_theo = m.group(1)
        ngay = ngay or _iso(m.group(2), m.group(3), m.group(4))
    can_cu = RE_CAN_CU.findall(header)
    return so, ngay, loai, trich_yeu, kem_theo, can_cu


def _body_start(lines: List[str]) -> int:
    for i, line in enumerate(lines):
        if RE_DIEU.match(line.strip()):
            return i
    # văn bản không chia Điều (thông báo, hướng dẫn, kế hoạch): bắt đầu ở mục "1."
    for i, line in enumerate(lines):
        if RE_MUC.match(line.strip()) and i > 3:
            return i
    return len(lines)


def parse_document(path: Path) -> tuple[Document, List[Passage], List[str]]:
    raw = unicodedata.normalize("NFC", path.read_text(encoding="utf-8"))
    lines = raw.splitlines()
    start = _body_start(lines)
    so, ngay, loai, trich_yeu, kem_theo, can_cu = _parse_header(raw, lines, start)
    doc_id = so or f"{DOC_TYPES.get(loai, loai)}|{trich_yeu}"
    hl = (_iso(*m.groups()) if (m := RE_HIEU_LUC.search(raw)) else None)
    doc = Document(doc_id=doc_id, file=path.name, loai=loai, so=so, trich_yeu=trich_yeu,
                   ngay_ban_hanh=ngay, hieu_luc=hl, kem_theo=kem_theo, raw=raw)

    passages: List[Passage] = []
    body = [l.strip() for l in lines[start:] if l.strip()]
    has_dieu = any(RE_DIEU.match(l) for l in body)
    if has_dieu:
        cur_dieu, cur_title, khoans, dieu_line = None, None, [], None

        def flush():
            if cur_dieu is None:
                return
            if khoans:
                for k, t in khoans:
                    passages.append(Passage(pid=f"{doc_id}#Đ{cur_dieu}.K{k}", doc_id=doc_id, dieu=cur_dieu,
                                            dieu_title=cur_title, khoan=k, unit="Khoản", text=t))
            else:
                passages.append(Passage(pid=f"{doc_id}#Đ{cur_dieu}", doc_id=doc_id, dieu=cur_dieu,
                                        dieu_title=None, khoan=None, unit="Điều", text=cur_title))

        for line in body:
            if m := RE_DIEU.match(line):
                flush()
                cur_dieu, cur_title, khoans = int(m.group(1)), m.group(2).strip(), []
            elif (m := RE_MUC.match(line)) and cur_dieu is not None:
                khoans.append((int(m.group(1)), m.group(2).strip()))
            elif khoans:
                khoans[-1] = (khoans[-1][0], khoans[-1][1] + " " + line)
        flush()
    else:
        for line in body:
            if m := RE_MUC.match(line):
                n = int(m.group(1))
                passages.append(Passage(pid=f"{doc_id}#M{n}", doc_id=doc_id, dieu=None, dieu_title=None,
                                        khoan=n, unit="Mục", text=m.group(2).strip()))
            elif passages:
                passages[-1].text += " " + line
    return doc, passages, can_cu


def _citation(doc: Document, p: Passage) -> str:
    if p.unit == "Khoản":
        return f"{doc.short}, Điều {p.dieu}, Khoản {p.khoan}"
    if p.unit == "Điều":
        return f"{doc.short}, Điều {p.dieu}"
    return f"{doc.short}, Mục {p.khoan}"


def _content(doc: Document, p: Passage) -> str:
    head = f"[{doc.short}" + (f" – {doc.trich_yeu}" if doc.so else "") + "]"
    if p.unit == "Khoản":
        loc = f"Điều {p.dieu}. {p.dieu_title} – Khoản {p.khoan}"
    elif p.unit == "Điều":
        loc = f"Điều {p.dieu}"
    else:
        loc = f"Mục {p.khoan}"
    return f"{head} {loc}: {p.text}"


def load_corpus(corpus_dir: Path, core_files: Optional[set] = None, reference_date: Optional[str] = None) -> Corpus:
    reference_date = reference_date or date.today().isoformat()
    docs: Dict[str, Document] = {}
    passages: Dict[str, Passage] = {}
    can_cu_map: Dict[str, List[str]] = {}
    for path in sorted(Path(corpus_dir).glob("*.txt")):
        doc, ps, can_cu = parse_document(path)
        doc.is_core = core_files is None or path.name in core_files
        docs[doc.doc_id] = doc
        can_cu_map[doc.doc_id] = can_cu
        for p in ps:
            p.citation = _citation(doc, p)
            p.content = _content(doc, p)
            passages[p.pid] = p
            doc.passages.append(p.pid)

    by_so = {d.so: d.doc_id for d in docs.values() if d.so}
    by_kem_theo = {d.kem_theo: d.doc_id for d in docs.values() if d.kem_theo}
    relations: List[Relation] = []

    def resolve_so(so: str) -> Optional[str]:
        return by_so.get(so)

    # Quy chế hiệu lực theo quyết định ban hành nó
    for d in docs.values():
        if d.kem_theo and not d.hieu_luc and d.kem_theo in by_so:
            d.hieu_luc = docs[by_so[d.kem_theo]].hieu_luc
        if not d.hieu_luc:
            d.hieu_luc = d.ngay_ban_hanh

    # 1) căn cứ (phần đầu văn bản) — tạo nút văn bản ngoài kho nếu chưa có
    for doc_id, lines in can_cu_map.items():
        for line in lines:
            for m in RE_SO_VB.finditer(line):
                kind_name, so = m.group(1), m.group(2)
                target = resolve_so(so)
                if target is None:
                    loai = kind_name.upper()
                    rest = line[m.end():]
                    rest = re.sub(r"^\s*ngày \d{1,2} tháng \d{1,2} năm \d{4}\s*", "", rest).strip()
                    ext = Document(doc_id=so, file=None, loai=loai, so=so, trich_yeu=rest,
                                   ngay_ban_hanh=(_iso(*mm.groups()) if (mm := RE_NGAY.search(line)) else None),
                                   hieu_luc=None, external=True, is_core=doc_id in {k for k, v in docs.items() if v.is_core})
                    docs[so] = ext
                    by_so[so] = so
                    target = so
                if target != doc_id:
                    relations.append(Relation("can_cu", doc_id, target, evidence=f"Căn cứ {line}"))

    # 2) ban hành kèm theo
    for d in docs.values():
        if d.kem_theo and d.kem_theo in by_so:
            relations.append(Relation("ban_hanh", d.doc_id, by_so[d.kem_theo],
                                      evidence=f"Ban hành kèm theo Quyết định số {d.kem_theo}"))

    def find_passage(doc_id: str, dieu: int, khoan: Optional[int]) -> Optional[str]:
        for pid in docs[doc_id].passages:
            p = passages[pid]
            if p.dieu == dieu and (khoan is None or p.khoan == khoan):
                return pid
        return None

    # 3) sửa đổi & 4) dẫn chiếu (trong thân văn bản)
    for d in list(docs.values()):
        if d.external:
            continue
        amend_target = None
        if m := RE_SUA_DOI_TARGET.search(d.trich_yeu):
            amend_target = by_kem_theo.get(m.group(1))
        for pid in d.passages:
            p = passages[pid]
            for m in RE_SUA_DOI.finditer(p.text):
                if amend_target is None:
                    continue
                k, dieu = int(m.group(1)), int(m.group(2))
                dst = find_passage(amend_target, dieu, k)
                relations.append(Relation("sua_doi", d.doc_id, amend_target, src_pid=pid, dst_pid=dst,
                                          evidence=m.group(0)))
            for m in RE_SO_VB.finditer(p.text):
                target = resolve_so(m.group(2))
                if target and target != d.doc_id:
                    relations.append(Relation("dan_chieu", d.doc_id, target, src_pid=pid, evidence=m.group(0)))
            for m in RE_HIEN_HANH.finditer(p.text):
                name = norm(f"{m.group(1)} {m.group(2)}")
                for t in docs.values():
                    if not t.external and norm(f"{t.loai} {t.trich_yeu}") == name and t.doc_id != d.doc_id:
                        relations.append(Relation("dan_chieu", d.doc_id, t.doc_id, src_pid=pid, evidence=m.group(0)))

    # Hiệu lực: đoạn bị sửa đổi KHÔNG bị xoá, chỉ gắn nhãn (và bị giảm điểm ở bước truy xuất)
    for r in relations:
        if r.kind == "sua_doi" and r.dst_pid:
            src_doc = docs[r.src_doc]
            if src_doc.hieu_luc and src_doc.hieu_luc <= reference_date:
                tgt = passages[r.dst_pid]
                src = passages[r.src_pid]
                tgt.superseded_by = r.src_pid
                hl = date.fromisoformat(src_doc.hieu_luc).strftime("%d/%m/%Y")
                tgt.superseded_note = f"đã bị sửa bởi {src.citation} (hiệu lực {hl})"

    # bỏ trùng
    seen, uniq = set(), []
    for r in relations:
        key = (r.kind, r.src_doc, r.dst_doc, r.src_pid, r.dst_pid)
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return Corpus(docs=docs, passages=passages, relations=uniq, reference_date=reference_date)
