# 🚀 LLM Analysis Quiz – Automated Solver

An automated multi-step quiz solver that renders JavaScript-powered pages, extracts data (CSV, PDF, HTML tables, scraped pages), analyzes them programmatically, and posts the correct answers back to the quiz server.  
Supports optional LLM-powered reasoning for ambiguous tasks.

## 📘 Overview

Server receives a POST request like:

```json
{
  "email": "your email",
  "secret": "your secret",
  "url": "https://example.com/quiz-834"
}
```

Solver must:

1. Validate inputs  
2. Open the provided quiz URL in a **headless browser**  
3. Understand the instructions from the webpage  
4. Download → clean → parse → analyze required data  
5. Submit the answer to the submit URL found on the page  
6. Follow the chain if a new quiz URL is returned  

This repository contains a fully automated engine that accomplishes this.

## ✨ Features

### 🕸 Web Scraping
- Full JavaScript rendering using **Playwright**
- Extracts from:
  - Dynamic DOM
  - `<pre>` embedded JSON
  - Relative & absolute URLs
  - Hidden submit links

### 📂 Data Extraction
- **CSV parsing** with:
  - Header detection
  - Numeric cleaning
  - Cutoff filtering
  - Column-name inference

- **PDF parsing** with:
  - Page-by-page table extraction
  - Value-column detection

- **HTML table parsing** via Pandas

### 🔍 Scrape Logic
Supports demo patterns like:
```
Scrape /demo-scrape-data?... → extract secret → POST to /submit
```

### 🤖 Optional LLM Module
Allows AI assistance for:
- OCR  
- Audio transcription  
- Answer extraction when unclear  
- Fallback interpretation of instructions  

Activate with:
```
ENABLE_LLM=1
OPENAI_API_KEY=your-key
```

### 🧩 Multi-Step Quiz Flow
Automatically follows:
```
quiz → submit → next quiz → submit → ... → end
```

## 📁 Project Structure

```
app/
 ├── main.py              # FastAPI entrypoint
 ├── solver.py            # Core multi-step solver
 ├── utils/
 │     └── pdf_utils.py   # PDF parsing helpers
 ├── requirements.txt
 └── README.md
```

## 🔧 Environment Variables

| Variable | Description |
|----------|-------------|
| `ENABLE_LLM` | Set to `1` to turn on optional LLM reasoning |
| `OPENAI_API_KEY` | API Key for LLM calls |
| `LLM_MODEL` | LLM model (default: `gpt-4o-mini`) |
| `PORT` | Required for Render deployment |

## 🧪 Testing With the Demo Quiz

### Sample PowerShell Test Script

```powershell
$endpoint = "https://<your-deployment-url>/quiz"

$payload = @{
  email  = "xxxxxxxxxxxxxxxxxxxxxxxxx"
  secret = "test-secret-123"
  url    = "https://tds-llm-analysis.s-anand.net/demo"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri $endpoint -ContentType "application/json" -Body $payload | ConvertTo-Json -Depth 10
```

If everything is correct, the output will have:
- Scrape task solved ✔  
- CSV task solved ✔  
- Final `"correct": true` response ✔  

## 📄 License

MIT License

