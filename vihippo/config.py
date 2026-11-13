"""Cấu hình chung cho demo ViHippo."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
OUTPUT_DIR = ROOT / "outputs" / "hipporag"  # đồ thị, embedding, OpenIE, cache LLM (sqlite)
CACHE_DIR = ROOT / "cache"                  # JSON đã tính sẵn cho UI (chạy được --offline)

# Ngày tham chiếu khi xét hiệu lực (mặc định: ngày chạy). Có thể cố định để demo ổn định.
REFERENCE_DATE = os.getenv("VIHIPPO_REFERENCE_DATE", "2026-11-13")

EMBEDDING_MODEL = os.getenv("VIHIPPO_EMBEDDING", "BAAI/bge-m3")

CORE_FILES = {
    "01_qd_100_ban_hanh_quy_che_ths.txt",
    "02_quy_che_dao_tao_ths.txt",
    "03_qd_245_sua_doi_quy_che_ths.txt",
    "04_tb_58_bao_ve_thang_1_2027.txt",
    "05_hd_12_chuan_ngoai_ngu.txt",
}

SAMPLE_QUESTIONS = [
    {
        "id": "q1_multihop",
        "label": "Đa bước",
        "question": "Hồ sơ bảo vệ đợt tháng 1/2027 phải nộp chậm nhất bao lâu trước ngày hội đồng họp?",
        "expected": "Không muộn hơn 15 ngày làm việc tính đến ngày họp hội đồng (QĐ 245 Điều 1 Khoản 1 sửa Khoản 2 Điều 5 Quy chế; TB 58 Mục 3 dẫn chiếu Quy chế hiện hành). Con số 30 ngày là quy định cũ.",
        "gold": ["245/QĐ-ĐHABC#Đ1.K1", "58/TB-ĐHABC#M3", "Quy chế|Đào tạo trình độ thạc sĩ#Đ1.K3"],
        "why": "Thông báo 58 không ghi thời hạn mà dẫn chiếu \"Quy chế hiện hành\"; Quy chế ghi 30 ngày trước ngày bảo vệ nhưng đã bị QĐ 245 sửa thành 15 ngày làm việc tính đến ngày họp hội đồng.",
    },
    {
        "id": "q2_amended",
        "label": "Bẫy văn bản cũ",
        "question": "Hội đồng đánh giá đề án gồm mấy thành viên?",
        "expected": "03 người (QĐ 245 sửa Khoản 1 Điều 7); con số 05 là quy định cũ",
        "gold": ["245/QĐ-ĐHABC#Đ1.K2"],
        "why": "Quy chế gốc ghi 05 thành viên, QĐ 245 sửa thành 03 người bằng từ ngữ khác hẳn.",
    },
    {
        "id": "q3_singlehop",
        "label": "Đơn bước",
        "question": "Học viên cao học cần đạt IELTS tối thiểu bao nhiêu để đạt chuẩn ngoại ngữ đầu ra?",
        "expected": "IELTS từ 5.5 trở lên (HD 12/HD-ĐHABC, Mục 2)",
        "gold": ["12/HD-ĐHABC#M2"],
        "why": "Câu trả lời nằm gọn trong một đoạn — RAG thường cũng phải làm được.",
    },
]

# Tham số truy xuất
TOP_K = 5  # = qa_top_k mặc định của HippoRAG 2 (paper §4.4: "top-5 retrieved passages"); đề cương dùng Recall@5
SUPERSEDED_FACTOR = float(os.getenv("VIHIPPO_SUPERSEDED_FACTOR", "0.5"))  # giảm điểm đoạn đã bị sửa
BELONG_WEIGHT = 0.5  # cạnh đoạn văn – văn bản chứa nó


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def is_offline() -> bool:
    return os.getenv("VIHIPPO_OFFLINE", "0") == "1"
