# BankSight API image. Build from the project root:  docker build -f docker/Dockerfile -t banksight-api .
FROM python:3.11-slim

# do not write .pyc files, print logs immediately, and let Python find our package in src/
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# install libraries first, so Docker can reuse this layer when only our code changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt

# our code and the models saved by the notebooks (models/ is large: it holds the RoBERTa folder)
COPY src/ src/
COPY models/ models/

EXPOSE 8000

# checks every 30 seconds that /health answers
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "banksight.api.main:app", "--host", "0.0.0.0", "--port", "8000"]