"""Prompt tiếng Việt thay cho prompt tiếng Anh mặc định của HippoRAG 2.

Ví dụ minh hoạ (one-shot / demo) dùng một trường hư cấu KHÁC ("Trường Cao đẳng XYZ")
để không lộ nội dung kho dữ liệu demo vào prompt.
"""
import json
from string import Template

# ---------------------------------------------------------------- OpenIE: NER
NER_SYSTEM = """Nhiệm vụ của bạn là trích xuất các thực thể có tên và cụm danh từ quan trọng từ một đoạn văn bản hành chính tiếng Việt.
Thực thể gồm: tên văn bản (kèm số, ký hiệu), điều/khoản được nhắc tới, đơn vị, chức danh, đối tượng (học viên, hội đồng…), mốc thời gian, thời hạn, con số, mức tiền.
Giữ nguyên chữ tiếng Việt có dấu. Trả về JSON dạng {"named_entities": [...]}."""

NER_EXAMPLE_PASSAGE = """[QĐ 77/QĐ-XYZ – Về việc ban hành Quy định thực tập] Điều 4. Thời gian thực tập – Khoản 1: Sinh viên thực tập tại doanh nghiệp tối thiểu 08 tuần và nộp báo cáo thực tập cho Khoa chậm nhất 10 ngày sau khi kết thúc đợt thực tập."""

NER_EXAMPLE_OUTPUT = """{"named_entities": ["QĐ 77/QĐ-XYZ", "Quy định thực tập", "Điều 4", "sinh viên", "doanh nghiệp", "08 tuần", "báo cáo thực tập", "Khoa", "10 ngày sau khi kết thúc đợt thực tập"]}"""

NER_TEMPLATE = [
    {"role": "system", "content": NER_SYSTEM},
    {"role": "user", "content": NER_EXAMPLE_PASSAGE},
    {"role": "assistant", "content": NER_EXAMPLE_OUTPUT},
    {"role": "user", "content": "${passage}"},
]

# ------------------------------------------------------- OpenIE: bộ ba (triple)
TRIPLE_SYSTEM = """Nhiệm vụ của bạn là xây dựng đồ thị tri thức từ một đoạn văn bản hành chính tiếng Việt và danh sách thực thể của đoạn đó.
Trả về JSON dạng {"triples": [[chủ thể, quan hệ, đối tượng], ...]}.

Yêu cầu:
- Mỗi bộ ba nên chứa ít nhất một, tốt nhất là hai, thực thể trong danh sách.
- Thay đại từ và cách nói tắt ("Quyết định này", "Quy chế", "khoản 2 Điều 5") bằng tên đầy đủ khi đoạn văn cho phép xác định.
- Quan hệ viết ngắn gọn bằng tiếng Việt (vd: "quy định thời hạn", "sửa đổi", "gồm", "có hiệu lực từ").
- Giữ nguyên con số, thời hạn, đơn vị đo như trong văn bản.
"""

TRIPLE_FRAME = """Chuyển đoạn văn sau thành JSON gồm danh sách bộ ba.
Đoạn văn:
```
{passage}
```

{named_entity_json}
"""

TRIPLE_EXAMPLE_OUTPUT = """{"triples": [
    ["QĐ 77/QĐ-XYZ", "ban hành", "Quy định thực tập"],
    ["Quy định thực tập", "có", "Điều 4"],
    ["sinh viên", "thực tập tại", "doanh nghiệp"],
    ["thời gian thực tập", "tối thiểu", "08 tuần"],
    ["sinh viên", "nộp", "báo cáo thực tập"],
    ["báo cáo thực tập", "nộp cho", "Khoa"],
    ["báo cáo thực tập", "có thời hạn nộp", "10 ngày sau khi kết thúc đợt thực tập"]
]}"""

TRIPLE_TEMPLATE = [
    {"role": "system", "content": TRIPLE_SYSTEM},
    {"role": "user", "content": TRIPLE_FRAME.format(passage=NER_EXAMPLE_PASSAGE, named_entity_json=NER_EXAMPLE_OUTPUT)},
    {"role": "assistant", "content": TRIPLE_EXAMPLE_OUTPUT},
    {"role": "user", "content": TRIPLE_FRAME.replace("{passage}", "${passage}").replace("{named_entity_json}", "${named_entity_json}")},
]

# ------------------------------------------------- NER cho câu hỏi (ner_query)
QUERY_NER_TEMPLATE = [
    {"role": "system", "content": "Bạn là hệ thống nhận diện thực thể trong câu hỏi tiếng Việt về văn bản hành chính."},
    {"role": "user", "content": "Hãy trích xuất mọi thực thể có tên/cụm danh từ quan trọng để trả lời câu hỏi dưới đây, trả về JSON.\n\nCâu hỏi: Sinh viên Trường Cao đẳng XYZ nộp báo cáo thực tập cho đơn vị nào?\n"},
    {"role": "assistant", "content": '{"named_entities": ["sinh viên", "Trường Cao đẳng XYZ", "báo cáo thực tập"]}'},
    {"role": "user", "content": "Câu hỏi: ${query}"},
]


def to_templates(chat):
    return [{"role": m["role"], "content": Template(m["content"])} for m in chat]


# ------------------------------------- Bộ lọc bộ ba (recognition memory) HippoRAG 2
FILTER_INSTRUCTIONS = (
    "Bạn là bộ lọc trong hệ thống hỏi đáp văn bản hành chính tiếng Việt. Cho một câu hỏi và danh sách bộ ba ứng viên "
    "(chủ thể, quan hệ, đối tượng) lấy từ kho văn bản, hãy giữ lại TỐI ĐA 4 bộ ba thật sự liên quan, có thể cần suy luận "
    "nhiều bước (ví dụ: thông báo dẫn chiếu tới quy chế, quy chế bị một quyết định sửa đổi). Chỉ dùng bộ ba có trong danh "
    "sách, không tự tạo bộ ba mới. Nếu không có bộ ba nào liên quan, trả về {\"fact\": []}. Sau đó, trong mục reason, "
    "giải thích ngắn gọn bằng tiếng Việt (mỗi bộ ba một dòng) vì sao giữ lại."
)

FILTER_SYSTEM = (
    "Các trường đầu vào:\n1. `question` (str): câu hỏi\n2. `fact_before_filter` (str): các bộ ba ứng viên\n\n"
    "Các trường đầu ra:\n1. `fact_after_filter` (Fact): các bộ ba được giữ, dạng JSON\n2. `reason` (str): lý do giữ từng bộ ba\n\n"
    "Mọi tương tác theo cấu trúc sau:\n\n[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n"
    "[[ ## fact_after_filter ## ]]\n{fact_after_filter}        # giá trị phải parse được theo JSON schema: "
    '{"type": "object", "properties": {"fact": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}}, "required": ["fact"]}'
    "\n\n[[ ## reason ## ]]\n{reason}\n\n[[ ## completed ## ]]\n\nMục tiêu: \n        " + FILTER_INSTRUCTIONS
)


def _demo(question, before, after, reasons):
    return {
        "question": question,
        "fact_before_filter": json.dumps({"fact": before}, ensure_ascii=False),
        "fact_after_filter": json.dumps({"fact": after}, ensure_ascii=False),
        "reason": "\n".join(f"- {json.dumps(f, ensure_ascii=False)}: {r}" for f, r in zip(after, reasons)),
    }


FILTER_DEMOS = [
    _demo(
        "Báo cáo thực tập đợt hè 2026 của Trường Cao đẳng XYZ phải nộp khi nào?",
        [["thông báo 15 tb xyz", "về", "đợt thực tập hè 2026"],
         ["sinh viên", "nộp báo cáo theo", "quy định thực tập hiện hành"],
         ["báo cáo thực tập", "có thời hạn nộp", "10 ngày sau khi kết thúc đợt thực tập"],
         ["sinh viên", "đóng", "học phí học kỳ hè"],
         ["thư viện", "mở cửa", "từ 7 giờ đến 21 giờ"]],
        [["thông báo 15 tb xyz", "về", "đợt thực tập hè 2026"],
         ["sinh viên", "nộp báo cáo theo", "quy định thực tập hiện hành"],
         ["báo cáo thực tập", "có thời hạn nộp", "10 ngày sau khi kết thúc đợt thực tập"]],
        ["xác định đúng đợt thực tập được hỏi",
         "thông báo không nêu ngày cụ thể mà dẫn chiếu sang quy định thực tập",
         "đây là thời hạn trong quy định được dẫn chiếu"],
    ),
    _demo(
        "Mức phụ cấp của giáo viên chủ nhiệm Trường Cao đẳng XYZ là bao nhiêu?",
        [["giáo viên chủ nhiệm", "được hưởng phụ cấp", "0 2 mức lương cơ sở"],
         ["qđ 90 qđ xyz", "sửa đổi", "mức phụ cấp giáo viên chủ nhiệm"],
         ["giáo viên chủ nhiệm", "được hưởng phụ cấp", "0 3 mức lương cơ sở"],
         ["sinh viên", "được miễn", "học phí"]],
        [["qđ 90 qđ xyz", "sửa đổi", "mức phụ cấp giáo viên chủ nhiệm"],
         ["giáo viên chủ nhiệm", "được hưởng phụ cấp", "0 3 mức lương cơ sở"],
         ["giáo viên chủ nhiệm", "được hưởng phụ cấp", "0 2 mức lương cơ sở"]],
        ["cho biết mức phụ cấp đã bị một quyết định sửa đổi",
         "mức phụ cấp có thể là mức mới sau sửa đổi",
         "mức phụ cấp có thể là mức cũ, cần giữ để đối chiếu"],
    ),
    _demo(
        "Thư viện Trường Cao đẳng XYZ có bao nhiêu đầu sách?",
        [["sinh viên", "nộp", "báo cáo thực tập"], ["giáo viên chủ nhiệm", "họp lớp", "mỗi tháng"]],
        [],
        [],
    ),
]

FILTER_DSPY = {
    "prog": {
        "lm": None, "traces": [], "train": [],
        "demos": FILTER_DEMOS,
        "signature": {"instructions": FILTER_INSTRUCTIONS, "fields": []},
        "system": FILTER_SYSTEM,
    }
}

FILTER_INPUT = (
    "[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n"
    "Hãy trả lời bằng các trường đầu ra, bắt đầu bằng `[[ ## fact_after_filter ## ]]` (JSON hợp lệ dạng Fact), "
    "tiếp theo là `[[ ## reason ## ]]`, và kết thúc bằng `[[ ## completed ## ]]`."
)
FILTER_OUTPUT = "[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## reason ## ]]\n{reason}\n\n[[ ## completed ## ]]"

# --------------------------------------------------------------- Trả lời (QA)
QA_SYSTEM = """Bạn là trợ lý tra cứu văn bản hành chính của một trường đại học. Quy tắc bắt buộc:
1. CHỈ dựa vào các đoạn văn bản được cung cấp, không dùng kiến thức bên ngoài.
2. Mỗi ý trong câu trả lời phải kèm trích dẫn đúng nhãn của đoạn, dạng [Văn bản, Điều, Khoản] hoặc [Văn bản, Mục].
3. Nếu một đoạn có ghi chú "ĐÃ BỊ SỬA ĐỔI", nội dung đó không còn áp dụng: dùng nội dung của văn bản sửa đổi và nói rõ văn bản nào đã sửa.
4. Nếu các đoạn không đủ căn cứ để trả lời, chỉ trả lời đúng một câu: "Không đủ căn cứ trong các văn bản được cung cấp để trả lời."
5. Trả lời ngắn gọn (tối đa 4 câu), bằng tiếng Việt."""


import re as _re
QA_SYSTEM_PARTIAL = _re.sub(
    r"^4\..*$",
    "4. Nếu các đoạn chỉ đủ căn cứ cho MỘT PHẦN câu hỏi: trả lời phần có căn cứ, rồi nói rõ điểm nào văn bản không quy định "
    "(không tự suy đoán). Nếu hoàn toàn không có căn cứ, trả lời đúng một câu: "
    "\"Không đủ căn cứ trong các văn bản được cung cấp để trả lời.\"",
    QA_SYSTEM, flags=_re.M)
assert QA_SYSTEM_PARTIAL != QA_SYSTEM

NO_RAG_SYSTEM = "Bạn là trợ lý trả lời câu hỏi về quy chế của Trường Đại học ABC. Trả lời ngắn gọn (tối đa 3 câu), bằng tiếng Việt."


def format_qa_user(question: str, passages: list) -> str:
    """passages: list of dict(citation, text, note)."""
    blocks = []
    for i, p in enumerate(passages, 1):
        note = f"\n(GHI CHÚ HIỆU LỰC: ĐÃ BỊ SỬA ĐỔI – {p['note']})" if p.get("note") else ""
        blocks.append(f"Đoạn {i} – nhãn trích dẫn: [{p['citation']}]\n{p['text']}{note}")
    return "Các đoạn văn bản:\n\n" + "\n\n".join(blocks) + f"\n\nCâu hỏi: {question}\nTrả lời:"
