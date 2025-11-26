# Dockerfile
FROM mcr.microsoft.com/playwright/python:latest

WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

EXPOSE 10000

ENV GUNICORN_CMD_ARGS="--timeout 120"

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:$PORT", "--workers", "1"]
