FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; \
    nltk.download('averaged_perceptron_tagger_eng', download_dir='/usr/share/nltk_data'); \
    nltk.download('averaged_perceptron_tagger', download_dir='/usr/share/nltk_data'); \
    nltk.download('cmudict', download_dir='/usr/share/nltk_data'); \
    nltk.download('punkt', download_dir='/usr/share/nltk_data'); \
    nltk.download('punkt_tab', download_dir='/usr/share/nltk_data')"

FROM python:3.11-slim AS runner
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /usr/share/nltk_data /usr/share/nltk_data
COPY --chown=appuser:appuser . /app

ENV PATH="/opt/venv/bin:$PATH"
ENV NLTK_DATA="/usr/share/nltk_data"

EXPOSE 8000

USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]