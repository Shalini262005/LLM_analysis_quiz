# app/solver.py
import time
import json
import re
import os
import tempfile
import logging
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
import pandas as pd
from playwright.sync_api import sync_playwright
from .utils.pdf_utils import extract_tables_from_pdf

logger = logging.getLogger(__name__)

# ---------------------------
# LLM helper / optional integration
# ---------------------------
ENABLE_LLM = os.getenv("ENABLE_LLM", "0") == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
# prefer openai package if installed
try:
    import openai  # type: ignore
    OPENAI_PY_AVAILABLE = True
except Exception:
    OPENAI_PY_AVAILABLE = False

def llm_available():
    return ENABLE_LLM and OPENAI_API_KEY is not None

def llm_call_simple(system_prompt: str, user_prompt: str, max_tokens: int = 512):
    """
    Make a simple LLM call and return assistant text.
    Uses openai package if available, otherwise uses httpx to call Chat Completions.
    This wrapper is intentionally minimal and meant for short textual tasks:
    - summarization
    - extract secret/value
    - answer extraction
    """
    if not llm_available():
        raise RuntimeError("LLM not available (enable with ENABLE_LLM=1 and set OPENAI_API_KEY).")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Prefer openai package if installed & configured
    if OPENAI_PY_AVAILABLE:
        try:
            openai.api_key = OPENAI_API_KEY
            resp = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0
            )
            return resp.choices[0].message["content"].strip()
        except Exception as e:
            logger.exception("openai package call failed: %s", e)
            # fallthrough to httpx approach
    # Fallback: httpx direct request to OpenAI-compatible endpoint (OpenAI API v1)
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
        r = httpx.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=60.0)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.exception("LLM httpx call failed: %s", e)
        raise

def detect_need_for_llm(page_text: str, page_html: str, links: list):
    """
    Very lightweight heuristics to decide if the page likely needs an LLM.
    Returns a dict with flags and reasons.
    """
    reasons = {"need": False, "why": []}
    if not page_text and not page_html:
        return reasons

    text = (page_text or "") + "\n" + (page_html or "")

    # audio detection
    if re.search(r"\.(mp3|wav|m4a|ogg)(\?|$)", ( " ".join(links) ).lower()) or re.search(r"\b(audio|listen|transcribe|speech|voice)\b", text, flags=re.IGNORECASE):
        reasons["need"] = True
        reasons["why"].append("transcription_audio")

    # image / ocr detection
    if re.search(r"\.(png|jpg|jpeg|tiff)(\?|$)", (" ".join(links)).lower()) or re.search(r"\b(image|photo|screenshot|ocr)\b", text, flags=re.IGNORECASE):
        reasons["need"] = True
        reasons["why"].append("ocr_image")

    # summarization / complex instruction keywords
    if re.search(r"\b(summarize|explain|interpret|visualize|plot|chart|transcribe|classify|predict)\b", text, flags=re.IGNORECASE):
        reasons["need"] = True
        reasons["why"].append("summarize_interpret")

    # secret / code extraction might be helped by LLM if ambiguous wording
    if re.search(r"\b(secret code|secret|code is|secret:|cutoff)\b", text, flags=re.IGNORECASE):
        # this is also solvable without LLM in many cases, but LLM may disambiguate
        reasons["need"] = True
        reasons["why"].append("extract_secret_possible")

    return reasons

def llm_extract_answer_from_text(body_text: str, question_hint: str = None):
    """
    Ask the LLM (if available) to extract an answer from the page text.
    question_hint can be used to instruct the LLM what to look for (e.g., 'find the cutoff' or 'extract secret code').
    Returns assistant text or None.
    """
    try:
        system = "You are a helpful assistant that extracts short machine-friendly answers from page text. Only return the answer, nothing else."
        user_prompt = "Page text:\n\n" + (body_text or "")
        if question_hint:
            user_prompt = question_hint + "\n\n" + user_prompt
        out = llm_call_simple(system, user_prompt, max_tokens=256)
        return out
    except Exception as e:
        logger.debug("llm_extract_answer error: %s", e)
        return None

# ---------------------------
# Existing helper functions (kept)
# ---------------------------
def find_submit_url_from_page(page):
    """
    Try to find a submit URL from form actions / anchors / data attributes.
    Returns the (possibly relative) URL string or None.
    """
    try:
        forms = page.query_selector_all("form")
        for f in forms:
            action = f.get_attribute("action")
            if action and "submit" in action:
                return action
        anchors = page.query_selector_all("a")
        for a in anchors:
            href = a.get_attribute("href")
            if href and ("submit" in href or "/submit" in href):
                return href
        el = page.query_selector("[data-submit]")
        if el:
            v = el.get_attribute("data-submit")
            if v:
                return v
        # fallback: search visible text for a submit URL
        body = page.inner_text("body") if page.query_selector("body") else ""
        m = re.search(r"(https?://[^\s'\"<>]*?/submit[^\s'\"<>]*)", body, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    except Exception as e:
        logger.debug("find_submit_url error: %s", e)
    return None

def is_numeric_series(series, threshold=0.6):
    s = series.astype(str).str.replace(r"[^0-9\.\-]", "", regex=True)
    parsed = pd.to_numeric(s, errors="coerce")
    non_null = parsed.notna().sum()
    total = len(parsed)
    if total == 0:
        return False
    return (non_null / total) >= threshold

def resolve_url(candidate: str, base: str):
    """
    Resolve a possibly-relative URL candidate against base.
    If candidate already absolute, return it unchanged.
    """
    if not candidate:
        return None
    candidate = candidate.strip()
    # If it already includes scheme, return as-is
    if bool(urlparse(candidate).scheme):
        return candidate
    if base:
        return urljoin(base, candidate)
    # As a last resort, if candidate looks like //host/path, add https:
    if candidate.startswith("//"):
        return "https:" + candidate
    return candidate

def try_submit_json(submit_url, payload, base_url=None, timeout=30.0):
    """
    POST JSON to submit_url, resolving relative URLs against base_url.
    Returns dict with status_code/body or error field.
    """
    try:
        final_url = resolve_url(submit_url, base_url) if base_url else submit_url
        if not final_url:
            return {"error": "no submit url provided"}
        r = httpx.post(final_url, json=payload, timeout=timeout)
        try:
            body = r.json()
        except Exception:
            body = r.text
        return {"status_code": r.status_code, "body": body, "posted_to": final_url}
    except Exception as e:
        return {"error": str(e), "attempted_url": submit_url, "resolved_url": (resolve_url(submit_url, base_url) if base_url else None)}

def extract_secret_from_text(text):
    """
    Heuristics to extract a 'secret' token from page text.
    Priority:
      1) Numeric token (3+ digits) appearing after phrases like 'secret', 'secret code', 'code is', etc.
      2) Longest numeric token on the page (>=3 digits).
      3) Alphanumeric token after the word 'secret' (fallback).
      4) First long alphanumeric token (fallback).
    Returns token string or None.
    """
    if not text:
        return None

    # Normalize whitespace
    txt = " ".join(text.split())

    # 1) look for patterns like "secret code is 21288" or "Secret: 21288"
    patterns = [
        r"(?:secret|secret code|the secret code|secret:|secret=)\s*(?:is|:|-)?\s*([0-9]{3,64})",
        r"(?:code|code is|code:)\s*([0-9]{3,64})",
        r"(?:secret|secret code|code)\s*(?:is|:|=)?\s*['\"]?([A-Za-z0-9_\-]{3,64})['\"]?"
    ]
    for pat in patterns:
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if m:
            token = m.group(1)
            if token:
                return token

    # 2) fallback: pick the longest numeric token (3+ digits) on the page
    numeric_tokens = re.findall(r"\d{3,64}", txt)
    if numeric_tokens:
        # choose the longest numeric token (likely the secret)
        numeric_tokens.sort(key=lambda s: -len(s))
        return numeric_tokens[0]

    # 3) fallback: alphanumeric token after 'secret' (if numeric not found)
    m2 = re.search(r"(?:secret|secret code|code)\s*(?:is|:|=)?\s*['\"]?([A-Za-z0-9_\-]{4,64})['\"]?", txt, flags=re.IGNORECASE)
    if m2:
        return m2.group(1)

    # 4) last resort: any long alphanumeric token
    tokens = re.findall(r"[A-Za-z0-9_\-]{6,64}", txt)
    for t in tokens:
        if not re.fullmatch(r"https?|http|submit|demo|page|email|secret|code", t, flags=re.IGNORECASE):
            return t

    return None

# ---------------------------
# Main solver (keeps your flow, with LLM enhancements)
# ---------------------------
def solve_quiz(email, secret, url, deadline):
    """
    Visit the URL, attempt to extract the task, compute an answer, and post it to the
    submit URL found on the page. Returns a result dict (with debug fields).
    """
    if time.time() > deadline:
        raise TimeoutError("Deadline already passed")

    result = {"start_url": url, "ts": time.time(), "llm": {"enabled": llm_available()}}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # navigate
        page.goto(url, wait_until="networkidle", timeout=30000)
        time.sleep(0.1)

        try:
            result["title"] = page.title()
        except Exception:
            result["title"] = None

        try:
            body_text = page.inner_text("body")
            result["body_preview"] = body_text[:2000]
        except Exception:
            body_text = None
            result["body_preview"] = None

        # collect anchors for detection
        anchors_texts = []
        try:
            anchors = page.query_selector_all("a")
            for a in anchors:
                try:
                    href = a.get_attribute("href")
                    text = a.inner_text()
                    anchors_texts.append(str(href or "") + " " + str(text or ""))
                except Exception:
                    continue
        except Exception:
            anchors_texts = []

        # Automatic LLM detection & optional attempt
        try:
            detection = detect_need_for_llm(body_text, page.content(), anchors_texts)
            result["llm_detection"] = detection
            if detection.get("need") and llm_available():
                # Attempt an LLM-based extraction first (non-destructive)
                try:
                    # If there's audio, ask LLM to say we need transcription (we handle audio later)
                    hint = "Extract a short machine readable answer from the following page text. If a numeric secret/cutoff is present return only that number. If the page instructs to download a CSV/AUDIO/IMAGE and compute an answer, say 'NEEDS_DOWNLOAD' and what to download (url or pattern)."
                    # include a short preview (avoid huge payloads)
                    preview_text = (body_text or "")[:4000]
                    llm_resp = llm_extract_answer_from_text(preview_text, question_hint=hint)
                    result["llm_quick_response"] = llm_resp
                    # Interpret LLM response heuristically
                    if llm_resp:
                        # if LLM returned a clean number -> treat as answer
                        m_num = re.search(r"\b([0-9]{2,64})\b", llm_resp)
                        if m_num:
                            candidate = m_num.group(1)
                            result["llm_extracted_numeric_candidate"] = candidate
                            payload = {"email": email, "secret": secret, "url": url, "answer": candidate}
                            if find_submit_url_from_page(page):
                                submit_url = find_submit_url_from_page(page)
                                submit_result = try_submit_json(submit_url, payload, base_url=url)
                                result["submit_result"] = submit_result
                                return result
                            # otherwise fall through to normal flow
                except Exception as e:
                    logger.debug("llm quick extraction failed: %s", e)
                    result.setdefault("llm_errors", []).append(str(e))
        except Exception as e:
            logger.debug("llm detection error: %s", e)

        # find submit url (may be relative)
        submit_url = find_submit_url_from_page(page)
        result["submit_url"] = submit_url

        # if not found, try to extract a /submit link from text
        if not submit_url and body_text:
            m = re.search(r"(\/[^\s'\"<>]*?submit[^\s'\"<>]*)", body_text, flags=re.IGNORECASE)
            if m:
                submit_url = m.group(1)
                result["submit_url_extracted_from_text"] = True
                result["submit_url"] = submit_url

        # parse JSON in <pre> if present
        parsed = None
        try:
            pre = page.query_selector("pre")
            if pre:
                pre_text = pre.inner_text()
                result["pre_preview"] = pre_text[:2000]
                try:
                    parsed = json.loads(pre_text)
                    result["pre_json"] = parsed
                except Exception:
                    parsed = None
        except Exception:
            parsed = None

        # Detect instruction to scrape a relative path (e.g., "Scrape /demo-scrape-data?...")
        try:
            if body_text:
                m_rel = re.search(r"(\/[^\s'\"<>]*demo-scrape-data[^\s'\"<>]*)", body_text, flags=re.IGNORECASE)
                if m_rel:
                    relpath = m_rel.group(1)
                    scrape_url = resolve_url(relpath, page.url)
                    result["detected_scrape_instruction"] = {"relpath": relpath, "scrape_url": scrape_url}
                    # Visit scrape_url
                    try:
                        page.goto(scrape_url, wait_until="networkidle", timeout=20000)
                        time.sleep(0.1)
                        scrape_text = page.inner_text("body") if page.query_selector("body") else ""
                        result["scrape_page_preview"] = scrape_text[:2000]
                        secret_code = extract_secret_from_text(scrape_text)
                        result["scraped_secret_candidate"] = secret_code
                        # If found, prepare payload and submit to submit_url (resolve relative)
                        if secret_code:
                            parsed_url = urlparse(scrape_url)
                            canonical_scrape_url = urlunparse(parsed_url._replace(query="", params="", fragment=""))

                            payload = {
                                "email": email,
                                "secret": secret,
                                # canonical URL WITHOUT query params
                                "url": canonical_scrape_url,
                                "answer": secret_code
                            }

                            logger.debug(
                                "Posting scraped secret payload to submit_url=%s resolved=%s payload=%s",
                                submit_url,
                                resolve_url(submit_url, scrape_url),
                                {"email": email, "url": canonical_scrape_url, "answer": secret_code}
                            )

                            if submit_url:
                                submit_result = try_submit_json(submit_url, payload, base_url=scrape_url)
                                result["submit_result"] = submit_result
                            else:
                                result["submit_result"] = {"error": "no submit_url_found"}
                            return result
                    except Exception as e:
                        result["scrape_error"] = str(e)
        except Exception as e:
            logger.debug("scrape-detection error: %s", e)

        # If a PDF link exists, try to download & parse PDF tables
        pdf_link = None
        try:
            anchors = page.query_selector_all("a")
            for a in anchors:
                href = a.get_attribute("href")
                if href and href.lower().endswith(".pdf"):
                    pdf_link = href
                    break
        except Exception:
            pass
        result["found_pdf_link"] = bool(pdf_link)

        if pdf_link:
            tmpdir = tempfile.mkdtemp()
            try:
                # resolve pdf link
                pdf_url = resolve_url(pdf_link, page.url)
                # download via httpx (avoids Playwright download complexity)
                r = httpx.get(pdf_url, timeout=30.0)
                pdf_path = os.path.join(tmpdir, "download.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(r.content)
                result["downloaded_pdf"] = pdf_path

                tables_by_page = extract_tables_from_pdf(pdf_path)
                result["pdf_tables_count"] = sum(len(dfs) for dfs in tables_by_page.values())

                # attempt to find "value" column on page 2
                answer_payload = None
                if 2 in tables_by_page:
                    for df in tables_by_page[2]:
                        cols_lower = [str(c).lower() for c in df.columns]
                        if "value" in cols_lower:
                            colname = df.columns[cols_lower.index("value")]
                            df[colname] = df[colname].astype(str).str.replace(r"[^0-9\.\-]", "", regex=True)
                            df = df[df[colname] != ""]
                            try:
                                df[colname] = pd.to_numeric(df[colname], errors="coerce")
                                total = float(df[colname].sum())
                                answer_payload = {
                                    "email": email,
                                    "secret": secret,
                                    "url": url,
                                    "answer": total
                                }
                                result["answer_reason"] = "sum_value_on_pdf_page_2"
                                result["answer"] = total
                                break
                            except Exception:
                                continue

                # fallback sums and submission
                if answer_payload and submit_url:
                    submit_result = try_submit_json(submit_url, answer_payload, base_url=url)
                    result["submit_result"] = submit_result
                return result

            except Exception as e:
                logger.exception("PDF processing error")
                result["pdf_error"] = str(e)
                return result

        # ---------------------------
        # NEW: handle CSV files mentioned on the page
        # ---------------------------
        try:
            # look for CSV links on the page
            csv_links = []
            try:
                anchors = page.query_selector_all("a")
                for a in anchors:
                    href = a.get_attribute("href")
                    type_attr = a.get_attribute("type") if a else None
                    text = a.inner_text() if a else ""
                    if not href:
                        continue
                    href_l = href.lower()
                    if href_l.endswith(".csv") or "csv" in (type_attr or "").lower() or ".csv?" in href_l:
                        csv_links.append(href)
                    # also catch links with CSV-like filenames in text
                    elif ".csv" in (text or "").lower():
                        csv_links.append(href)
            except Exception:
                csv_links = []

            # also check for "CSV file" instruction and a relative endpoint
            if not csv_links and body_text:
                # try to find e.g. "/demo-audio-data?..." patterns
                m_csv_rel = re.search(r"(\/[^\s'\"<>]*?(?:csv|demo-audio-data)[^\s'\"<>]*)", body_text, flags=re.IGNORECASE)
                if m_csv_rel:
                    csv_links.append(m_csv_rel.group(1))

            if csv_links:
                # Decide which CSV to use; prefer absolute or first one
                chosen = csv_links[0]
                csv_url = resolve_url(chosen, page.url)
                result["detected_csv_links"] = csv_links
                result["resolved_csv_url"] = csv_url

                # attempt to extract cutoff from the page text if present
                cutoff = None
                if body_text:
                    m_cut = re.search(r"Cutoff[:\s]*([0-9]+)", body_text, flags=re.IGNORECASE)
                    if m_cut:
                        cutoff = float(m_cut.group(1))
                        result["detected_cutoff"] = cutoff

                # download csv
                try:
                    r = httpx.get(csv_url, timeout=30.0)
                    r.raise_for_status()
                    tmpdir = tempfile.mkdtemp()
                    csv_path = os.path.join(tmpdir, "data.csv")
                    with open(csv_path, "wb") as f:
                        f.write(r.content)
                    result["downloaded_csv"] = csv_path

                    # read csv into pandas
                    try:
                        df = pd.read_csv(csv_path)
                        result["csv_shape"] = df.shape
                        # find numeric columns
                        numeric_cols = [c for c in df.columns if is_numeric_series(df[c])]
                        result["csv_numeric_columns"] = numeric_cols

                        answer_value = None
                        answer_reason = None

                        if numeric_cols:
                            first_col = numeric_cols[0]
                            # coerce to numeric
                            df[first_col] = pd.to_numeric(df[first_col].astype(str).str.replace(r"[^0-9\.\-]", "", regex=True), errors="coerce")
                            if cutoff is not None:
                                mask = df[first_col] >= cutoff
                                # sum values >= cutoff
                                answer_value = float(df.loc[mask, first_col].sum(min_count=1) or 0.0)
                                answer_reason = f"sum_{first_col}_>=_cutoff"
                            else:
                                # no cutoff: sum the column
                                answer_value = float(df[first_col].sum(min_count=1) or 0.0)
                                answer_reason = f"sum_{first_col}"
                        else:
                            # fallback: count rows
                            answer_value = int(len(df))
                            answer_reason = "row_count"

                        result["csv_answer"] = answer_value
                        result["csv_answer_reason"] = answer_reason

                        # build payload using the task's start_url (as other steps do)
                        submit_payload = {
                            "email": email,
                            "secret": secret,
                            "url": result.get("start_url") or url,
                            "answer": answer_value
                        }

                        # keep a preview (mask secret)
                        preview = dict(submit_payload)
                        preview["secret"] = "***"
                        result["submit_payload_preview"] = preview

                        if submit_url:
                            submit_result = try_submit_json(submit_url, submit_payload, base_url=csv_url)
                            result["submit_result"] = submit_result
                        else:
                            result["submit_result"] = {"error": "no submit_url_found"}
                        return result
                    except Exception as e:
                        result["csv_read_error"] = str(e)
                except Exception as e:
                    result["csv_download_error"] = str(e)
        except Exception as e:
            logger.debug("csv-detection error: %s", e)

        # Try reading HTML tables in the loaded page
        try:
            html = page.content()
            dfs = pd.read_html(html)
            result["html_table_count"] = len(dfs)
            answer_payload = None
            for df in dfs:
                numeric_cols = [c for c in df.columns if is_numeric_series(df[c])]
                if numeric_cols:
                    col = numeric_cols[0]
                    total = float(pd.to_numeric(df[col], errors="coerce").sum())
                    answer_payload = {
                        "email": email,
                        "secret": secret,
                        "url": url,
                        "answer": total
                    }
                    result["answer"] = total
                    result["answer_reason"] = "sum_first_numeric_col_html_table"
                    break
            if answer_payload:
                if submit_url:
                    submit_result = try_submit_json(submit_url, answer_payload, base_url=url)
                    result["submit_result"] = submit_result
                else:
                    result["submit_result"] = {"error": "no submit_url_found"}
                return result
        except Exception:
            # no html tables or parsing failed -> continue
            pass

        # If page contains a pre JSON template, fill it (overwrite placeholders) and submit
        if parsed is not None:
            try:
                parsed["email"] = email
                parsed["secret"] = secret
                parsed["url"] = url

                if "answer" in parsed:
                    if parsed["answer"] is None or str(parsed["answer"]).strip() == "" or "anything" in str(parsed["answer"]).lower():
                        parsed["answer"] = "local-test"
                else:
                    parsed["answer"] = "local-test"

                if submit_url:
                    submit_result = try_submit_json(submit_url, parsed, base_url=url)
                    result["submit_result"] = submit_result
                    return result
                else:
                    result["pre_parsed_but_no_submit_url"] = True
            except Exception as e:
                result["pre_submit_error"] = str(e)

        # As a fallback, check if body_text instructs to POST a value present on the page
        try:
            if body_text:
                # find relative endpoints to scrape (links) and try them
                m2 = re.search(r"(https?://[^\s'\"<>]*?(submit|answer)[^\s'\"<>]*)", body_text, flags=re.IGNORECASE)
                if m2:
                    candidate = m2.group(1)
                    result["fallback_submit_candidate"] = candidate
                    if parsed and isinstance(parsed, dict):
                        submit_result = try_submit_json(candidate, parsed, base_url=url)
                        result["submit_result"] = submit_result
                        return result

            # If nothing else worked, return what we have
            result["note"] = "no actionable answer found on this page"
            return result
        except Exception as e:
            result["fallback_error"] = str(e)
            return result
