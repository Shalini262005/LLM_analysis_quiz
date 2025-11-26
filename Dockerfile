# Dockerfile - use Playwright python base image (includes browsers & native libs)
FROM mcr.microsoft.com/playwright/python:latest

WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

EXPOSE 10000

ENV GUNICORN_CMD_ARGS="--timeout 120"

# Use shell form so $PORT is expanded from environment at container runtime.
CMD /bin/sh -c "exec gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:${PORT:-10000} --workers 1"
