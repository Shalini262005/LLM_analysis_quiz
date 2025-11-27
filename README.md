# LLM Analysis Quiz Solver — Render-ready reference implementation

## This FastAPI service:
- accepts quiz tasks
- renders JS pages with Playwright
- downloads PDFs/tables
- computes simple answers
- POSTs them to the quiz submit URL

###############################################
## Quick local / Render notes
###############################################

### Build command (Render or local)
pip install -r requirements.txt && python -m playwright install --with-deps

### Start command (Render)
gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:$PORT --workers 1

###############################################
## Environment variables (set these in Render)
###############################################

**QUIZ_SECRET  - your secret string (must match Google Form)**
**TIME_BUDGET  - optional, default 180 seconds**
**LOG_LEVEL    - optional (e.g., INFO)**

###############################################
## Test using the demo
###############################################

curl -X POST https://<your-render-url>/quiz \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","secret":"<YOUR_SECRET>","url":"https://tds-llm-analysis.s-anand.net/demo"}'

###############################################
## Notes
###############################################

- Keep QUIZ_SECRET secret — do NOT commit it.
- Ensure Playwright dependencies install:
     python -m playwright install --with-deps
- Render filesystem is ephemeral; temp files go to system temp dirs.
