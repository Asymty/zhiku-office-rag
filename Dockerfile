FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
    INKATLAS_DATABASE_PATH=/data/atlas.sqlite3

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=2 update \
    && apt-get -o Acquire::Retries=2 install -y --no-install-recommends libgomp1 tesseract-ocr-eng tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 zhiku \
    && useradd --uid 10001 --gid zhiku --create-home zhiku \
    && install -d -o zhiku -g zhiku /data

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

COPY app.py create_admin.py evaluate_rag.py compare_retrieval.py knowledge_base_tool.py LICENSE ./
COPY zhiku ./zhiku
COPY web ./web
COPY evaluation ./evaluation

USER zhiku
EXPOSE 8010
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8010"]
