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
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import QUIZ_SECRET, TIME_BUDGET, LOG_LEVEL
from .solver import solve_quiz

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Analysis Quiz Solver")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Return 400 for invalid JSON / validation errors to match project spec.
    FastAPI's default is 422 — the spec requires 400 for invalid JSON payloads.
    """
    return JSONResponse(
        status_code=400,
        content={"ok": False, "error": "invalid json", "details": str(exc)},
    )


# Ensure playwright browsers in container/runtime if missing (safety-net).
@app.on_event("startup")
def ensure_playwright_browsers():
    try:
        browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/tmp/ms-playwright")
        logger.info("PLAYWRIGHT_BROWSERS_PATH=%s", browsers_path)

        def chromium_exists(path: str) -> bool:
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
    """
    Receive a quiz POST, validate secret, then attempt to solve the provided URL.
    If the submit response returns a new 'url', follow it and attempt to solve that too,
    all within TIME_BUDGET seconds starting from the first request.
    Returns a trace of each step attempted.
    """
    # Secret check
    if payload.secret != QUIZ_SECRET:
        raise HTTPException(status_code=403, detail="invalid secret")

    deadline = time.time() + TIME_BUDGET
    results_trace: List[Dict[str, Any]] = []
    current_url: Optional[str] = payload.url

    try:
        # Loop while we have a url and time remains
        while current_url and time.time() < deadline:
            # solve_quiz visits the url, tries to compute answer and submit if possible
            res = solve_quiz(payload.email, payload.secret, current_url, deadline)
            results_trace.append(res)

            # Inspect submit_result to find a next url (if provided)
            submit_res = res.get("submit_result")
            next_url = None

            if isinstance(submit_res, dict):
                # submit_result may contain a status_code and body (which may be dict)
                body = submit_res.get("body")
                if isinstance(body, dict):
                    # Standard format uses "url" field for the next task
                    next_url = body.get("url")
                # Sometimes the solver may store direct 'url' in other keys
                if not next_url:
                    next_url = submit_res.get("url")

            # If no next url provided, stop the loop
            if next_url:
                current_url = next_url
                # continue loop (deadline check at top)
            else:
                break

        final_result = results_trace[-1] if results_trace else {}
    except Exception as e:
        logger.exception("Error solving quiz")
        return {"ok": False, "error": str(e), "trace": results_trace}

    return {"ok": True, "result": final_result, "trace": results_trace}


@app.get("/health")
def health():
    return {"status": "ok"}
