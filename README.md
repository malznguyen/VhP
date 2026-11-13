# ViHippo – bản demo

Truy xuất đa bước trên văn bản hành chính tiếng Việt, dựa trên **HippoRAG 2** (repo chính chủ
`OSU-NLP-Group/HippoRAG`, commit `d5c8329`, không sửa mã thư viện).

> Bản demo trên dữ liệu minh hoạ hư cấu ("Trường Đại học ABC"), chưa phải ViHippoRAG đầy đủ.

## Cài đặt (Windows, Python 3.12)

```powershell
git clone --depth 1 https://github.com/OSU-NLP-Group/HippoRAG.git third_party/HippoRAG
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e third_party/HippoRAG -r requirements.txt
copy .env.example .env   # điền Azure OpenAI: endpoint, key, deployment, api-version
```

Embedding `BAAI/bge-m3` chạy local trên CPU (tự tải ~2,3 GB ở lần đầu).

## Chạy

| Việc | Lệnh (có `make`) | Lệnh trực tiếp |
|---|---|---|
| Lập chỉ mục + cache 3 câu mẫu (cần mạng) | `make index` | `.venv\Scripts\python build_index.py --quiet` |
| Chỉ chạy RAG thường cho câu mẫu | `make baseline` | `... build_index.py --baseline --quiet` |
| Mở giao diện | `make demo` | `.venv\Scripts\python app.py` |
| Mở giao diện không cần mạng | `make offline` | `.venv\Scripts\python app.py --offline` |

- Kết quả OpenIE, embedding, đồ thị và **mọi lời gọi LLM** được cache trong `outputs/hipporag/`
  (cache sqlite của HippoRAG). Lúc demo không cần gọi lại LLM để lập chỉ mục.
- `cache/*.json` chứa sẵn mọi thứ UI cần cho 3 câu mẫu: các bước (a)–(e), mọi biến thể câu trả lời.
  `--offline` chỉ đọc các file này, không nạp bge-m3, không gọi mạng; chỉ ô hỏi tự do bị tắt.
- Máy không có `make` thì dùng cột lệnh trực tiếp.

## Cấu trúc

```
data/corpus/            19 văn bản hư cấu (5 lõi + 14 gây nhiễu) → 96 đoạn Điều/Khoản
vihippo/corpus.py       regex: tách Điều/Khoản/Mục, số ký hiệu, ngày, hiệu lực, căn cứ/sửa đổi/dẫn chiếu
vihippo/prompts_vi.py   prompt tiếng Việt: NER + bộ ba (OpenIE), NER câu hỏi, bộ lọc bộ ba, trả lời
vihippo/engine.py       lớp con ViHippoRAG(HippoRAG): nút văn bản + cạnh có trọng số, hiệu lực, vết (a)–(e)
build_index.py          lập chỉ mục, tính sẵn câu mẫu, ablation
app.py                  giao diện Streamlit
```

## Kịch bản trình bày 5 phút

1. **(0:00–0:45) Vì sao cần RAG.** “LLM giống người giỏi tiếng Việt nhưng chưa đọc quy chế trường.”
   Chỉ cột trái (hỏi thẳng: không biết/đoán) và cột phải (có trích dẫn [văn bản, Điều, Khoản]).
2. **(0:45–1:30) Kho văn bản.** 6 thẻ: QĐ 100 ban hành Quy chế; Quy chế ghi 30 ngày, 05 thành viên;
   QĐ 245 sửa hai khoản đó bằng từ ngữ khác; TB 58 chỉ ghi “theo Quy chế hiện hành”. Nhấn mạnh:
   quan hệ rút bằng regex từ thể thức văn bản, không dùng LLM. Thêm 14 văn bản nhiễu cùng trường.
3. **(1:30–2:15) Lập chỉ mục.** Chọn Quy chế → Điều 5 Khoản 2 → xem bộ ba → đồ thị thu gọn:
   chấm xanh lá là cụm từ, chấm xanh dương là đoạn, ô vuông nâu là văn bản, đường đỏ đậm là “sửa đổi”.
4. **(2:15–3:30) Hỏi thử và so sánh (mặc định top-5, như HippoRAG 2).** Chọn câu đa bước “đợt 1/2027”:
   RAG thường **từ chối** (5 đoạn giống chữ nhất không chứa QĐ 245 Đ1 K1, nó ở hạng 10), ViHippo trả lời
   **15 ngày làm việc trước ngày họp hội đồng**, dẫn QĐ 245 và nói rõ con số 30 ngày đã bị sửa.
   **Ví dụ “lấy nhầm văn bản na ná”:** bật “Cho phép trả lời một phần” rồi nhìn cột RAG thường: nó trả lời
   **45 ngày**, vì trong top-5 có Quy chế **tiến sĩ** (“nộp hồ sơ… chậm nhất 45 ngày trước ngày dự kiến bảo vệ”).
   Tìm theo độ giống chữ dễ lấy nhầm văn bản có câu chữ na ná nhưng khác đối tượng.
   Chọn câu hội đồng và câu IELTS: ở top-5 **cả hai hệ đều đúng** (03 thành viên; 5.5). Nói thẳng: với câu
   hội đồng, RAG thường chỉ sai khi lấy 3 đoạn (đổi sang top-3 thì nó trả lời 05, văn bản cũ).
5. **(3:30–4:30) Bên trong ViHippo.** Với câu đa bước: (a) bộ ba khớp → (b) LLM giữ bộ ba nào, vì sao →
   (c) hạt giống → kéo thanh trượt vòng 0 → 12 để thấy “mực loang” từ TB 58 sang Quy chế và QĐ 245 → (e) top-5.
6. **(4:30–5:00) Hạn chế (nói thật).** Mở “Gỡ dần từng thành phần”: **HippoRAG 2 gốc ở top-5 cũng trả lời đúng
   cả 3 câu**. Phần thắng RAG thường ở câu đa bước đến từ đồ thị + PPR của HippoRAG 2, chưa phải từ phần ViHippo
   thêm vào. Lớp nút văn bản & cạnh quan hệ còn đẩy QĐ 245 Đ1 K1 từ hạng 3 xuống hạng 4 (khung “Vì sao PPR chạy sai”:
   cụm từ ngày tháng hiếm kéo điểm, cạnh sửa đổi bị chia loãng). “Kế thừa điểm” đưa nó lên hạng 2: đó là ý tưởng
   cải tiến, chưa phải kết quả. Ba câu hư cấu không đủ để kết luận; luận văn sẽ đo Recall@5 trên bộ dữ liệu thật.

## Kết quả hiện tại trên 3 câu mẫu (đã cache, bản đóng băng)

Cấu hình mặc định: **top-5** cho cả hai hệ, trả lời chặt, không kế thừa điểm.
“HippoRAG 2 gốc” = cùng chỉ mục, bỏ nút văn bản, cạnh quan hệ và nhãn hiệu lực; cùng prompt trả lời chặt, top-5.

| Câu | Biến thể | RAG thường | HippoRAG 2 gốc | ViHippo |
|---|---|---|---|---|
| Hồ sơ đợt 1/2027 nộp chậm nhất bao lâu trước ngày hội đồng họp? | **Mặc định (top-5)** | **Từ chối** | **15 ngày làm việc** | **15 ngày làm việc**, dẫn QĐ 245 |
| | Top-5, trả lời một phần | 45 ngày (quy chế tiến sĩ) | — | 15 ngày làm việc |
| | Top-3 | Từ chối / 45 ngày (một phần) | — | Từ chối / “không xác định, theo Quy chế hiện hành” |
| | Kế thừa điểm (top-3 hoặc top-5) | — | — | 15 ngày làm việc |
| Hội đồng gồm mấy thành viên? | **Mặc định (top-5)** | **03** | **03** | **03** |
| | Top-3 | **05** (sai, văn bản cũ) | — | 03 |
| IELTS tối thiểu? | Mọi biến thể | 5.5 | 5.5 | 5.5 |

Hạng của đoạn cần tìm (1 = đầu bảng):

| Đoạn | RAG thường | HippoRAG 2 gốc | + nút văn bản & cạnh quan hệ (= ViHippo mặc định) | + kế thừa điểm |
|---|---|---|---|---|
| QĐ 245 Đ1 K1 (15 ngày làm việc) | 10 | 3 | 4 | 2 |
| TB 58 Mục 3 (“theo Quy chế hiện hành”) | 6 | 4 | 3 | 4 |
| Quy chế Đ1 K3 (định nghĩa ngày bảo vệ) | 7 | 16 | 16 | 16 |
| QĐ 245 Đ1 K2 (03 người) | 5 | 2 | 3 | 1 |
| HD 12 Mục 2 (IELTS 5.5) | ≤ 3 | 3 | 3 | 3 |

Ghi chú trung thực:
- Ở top-5, ViHippo **không hơn HippoRAG 2 gốc** ở câu nào trong 3 câu mẫu. So với RAG thường, ViHippo chỉ hơn ở câu
  đa bước. Ở top-3, ViHippo hơn RAG thường ở câu hội đồng, nhưng câu đa bước thì cả hai từ chối.
- Lớp nút văn bản & cạnh quan hệ theo thiết kế hiện tại làm **tụt** hạng đoạn sửa đổi (3 → 4 ở câu đa bước,
  2 → 3 ở câu hội đồng). Nhãn “đã bị sửa” không phải lý do HippoRAG 2 gốc trả lời đúng: chính đoạn QĐ 245 ghi
  “sửa đổi Khoản 2 Điều 5…”, nên LLM tự nhận ra quy định cũ.
- Khoản định nghĩa Quy chế Đ1 K3 không vào top-5 của hệ nào, không được dùng trong câu trả lời nào.
- Câu tự do “Thời gian đào tạo tối đa của chương trình thạc sĩ?” (thử khi còn 95 đoạn, top-3): RAG thường đúng (4 năm),
  ViHippo từ chối vì PPR kéo Công văn 412 (dẫn chiếu Quy chế) lên. Lúc trình bày chỉ dùng 3 câu mẫu.
- Câu trả lời do LLM suy luận (gpt-5.6-luna, không cố định temperature) sinh ra và được cache theo nội dung prompt.
  Chạy lại từ đầu (`--rebuild`) có thể cho câu chữ khác.

## Nhật ký điều chỉnh demo

Mọi thay đổi về dữ liệu, câu hỏi và cách chạy kể từ bản đầu (13/11/2026), theo thứ tự thời gian.

| # | Thay đổi | Lý do | Ảnh hưởng tới kết quả |
|---|---|---|---|
| 0 | Bản đầu: 19 văn bản hư cấu (5 lõi + 14 nhiễu), 95 đoạn. Câu đa bước: “…chậm nhất bao lâu trước **ngày bảo vệ**?” | Theo đặc tả. QĐ 245 cố ý viết bằng từ ngữ khác (“15 ngày làm việc tính đến ngày họp hội đồng”, “hội đồng chấm có 03 người”) | RAG thường kiểm tra trước khi làm UI: đa bước từ chối, hội đồng trả lời 05. Không chỉnh dữ liệu để làm hỏng baseline |
| 1 | Thêm tuỳ chọn “kế thừa điểm” (mặc định **tắt**) | PPR không đưa QĐ 245 vào top-3 ở câu đa bước; ablation cho thấy lớp quan hệ không giúp | Không đổi kết quả mặc định; chỉ là biến thể để so sánh |
| 2 | Thêm tuỳ chọn “cho phép trả lời một phần” và top-5 (mặc định **tắt**, top-3) | Kể cả khi có đủ đoạn, LLM vẫn từ chối vì “ngày họp hội đồng” ≠ “ngày bảo vệ” trong văn bản | Không đổi kết quả mặc định |
| 3 | Thêm Quy chế thạc sĩ Điều 1 Khoản 3: “Ngày bảo vệ đề án là ngày hội đồng họp đánh giá đề án.” TB 58 **không** sửa | Bổ sung mắt xích còn thiếu giữa “ngày bảo vệ” và “ngày họp hội đồng” | Không đổi câu trả lời nào: khoản này hạng 10 (RAG thường) và 17 (ViHippo), không vào top-3/top-5 |
| 4 | Thêm khung “Vì sao PPR chạy sai” ở trang câu đa bước | Giải thích thất bại thay vì che đi | Chỉ hiển thị, số liệu tính từ đồ thị/cache |
| 5 | Đổi câu đa bước thành “…chậm nhất bao lâu trước **ngày hội đồng họp**?”; cập nhật đáp án mong đợi | Khoản định nghĩa (#3) không được truy xuất; người hỏi dùng đúng mốc thời gian mà quyết định sửa đổi dùng | Câu hỏi giờ gần chữ của QĐ 245 hơn, có lợi cho cả hai hệ thống. Mặc định vẫn cả hai từ chối; ViHippo đúng khi bật kế thừa điểm hoặc top-5. Sau đó **đóng băng**, không chỉnh thêm |
| 6 | Đổi top-k mặc định từ 3 lên **5 cho cả hai hệ** (RAG thường và ViHippo); vẫn chọn được top-3. Thêm câu trả lời của HippoRAG 2 gốc ở top-5 để so sánh | Đề cương đánh giá truy xuất bằng Recall@5. HippoRAG 2 đọc top-5: paper (arXiv 2502.14802, mục 4.4) ghi “Our QA module uses the top-5 retrieved passages as context for an LLM”, mục 4.3 dùng “passage recall@5”; repo `src/hipporag/utils/config_utils.py` dòng 203–205: `qa_top_k` mặc định 5, dùng ở `HippoRAG.py` dòng 1141 (commit `d5c8329`). **Đổi sau khi đã thấy kết quả top-3**, nhưng các lý do này có từ trước (đề cương, paper) và top-3 lúc đầu là lựa chọn của đặc tả demo, không dựa trên nguồn nào | Câu đa bước: RAG thường vẫn từ chối, ViHippo đúng (trước đây cả hai từ chối). Câu hội đồng: RAG thường đúng (trước đây sai), nên ViHippo **mất** lợi thế ở câu này. HippoRAG 2 gốc top-5 đúng cả 3 câu. Không đổi dữ liệu, câu hỏi hay tuỳ chọn nào khác. Sau đó **đóng băng lại** |

Các thay đổi chỉ về hiển thị (không ảnh hưởng kết quả): viết tắt tên văn bản ngoài kho (“Thông tư”, “Nghị quyết”),
thu gọn đồ thị ở trang Lập chỉ mục, thanh bên tự gập trên điện thoại, trang hiện tại lưu trong URL (`?p=`),
dòng chú thích trong khung “Vì sao PPR chạy sai” chuyển từ câu viết tay sang câu trả lời thật lấy từ cache
(câu viết tay cũ không còn đúng sau thay đổi #5). Văn bản nhiễu và các văn bản lõi khác không bị sửa sau lần lập chỉ mục đầu.
