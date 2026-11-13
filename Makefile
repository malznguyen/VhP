PY ?= .venv/Scripts/python.exe

.PHONY: setup index baseline demo offline rebuild

setup:            ## tạo môi trường + cài HippoRAG 2 (repo chính chủ) và thư viện demo
	uv venv .venv --python 3.12
	uv pip install --python $(PY) -e third_party/HippoRAG -r requirements.txt

index:            ## lập chỉ mục + cache 3 câu mẫu (cần mạng, gọi Azure OpenAI lần đầu)
	$(PY) build_index.py --quiet

baseline:         ## chỉ chạy RAG thường với các câu mẫu
	$(PY) build_index.py --baseline --quiet

rebuild:          ## xoá chỉ mục và làm lại từ đầu
	$(PY) build_index.py --rebuild --quiet

demo:             ## mở giao diện
	$(PY) app.py

offline:          ## mở giao diện không cần mạng (chỉ dùng cache)
	$(PY) app.py --offline
