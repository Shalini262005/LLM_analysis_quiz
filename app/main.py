# app/main.py
import sys
if sys.platform == "win32":
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

import os
import subprocess
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .config import QUIZ_SECRET, TIME_BUDGET, LOG_LEVEL
from .solver import solve_quiz

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Analysis Quiz Solver")

# Ensure playwright browsers in container/runtime if missing (safety-net).
@app.on_event("startup")
def ensure_playwright_browsers():
    try:
        browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/tmp/ms-playwright")
        logger.info("PLAYWRIGHT_BROWSERS_PATH=%s", browsers_path)

        def chromium_exists(path):
            if not path or not os.path.isdir(path):
                return False
            for root, dirs, files in os.walk(path):
                if "chrome" in files or "chrome-linux" in root or "chromium" in files:
                    return True
            return False

        if not chromium_exists(browsers_path):
            logger.info("Chromium missing; attempting install to %s", browsers_path)
            env = os.environ.copy()
            env["PLAYWRIGHT_BROWSERS_PATH"] = browsers_path
            # best-effort install; in Docker base it will be quick or no-op
            subprocess.run([os.sys.executable, "-m", "playwright", "install", "chromium"], check=False, env=env)
            if chromium_exists(browsers_path):
                logger.info("Chromium installed successfully to %s", browsers_path)
            else:
                logger.warning("Chromium still not present at %s after install attempt", browsers_path)
        else:
            logger.info("Chromium already exists at %s", browsers_path)
    except Exception:
        logger.exception("Error ensuring Playwright browsers (startup)")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

@app.post("/quiz")
def receive_quiz(payload: QuizRequest):
    if payload.secret != QUIZ_SECRET:
        raise HTTPException(status_code=403, detail="invalid secret")
    deadline = time.time() + TIME_BUDGET
    try:
        result = solve_quiz(payload.email, payload.secret, payload.url, deadline)
    except Exception as e:
        logger.exception("Error solving quiz")
        return {"ok": False, "error": str(e)}
    return {"ok": True, "result": result}

@app.get("/health")
def health():
    return {"status": "ok"}
