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
    s = series.astype(str).str.replace(r"[^0-9\.\-eE\+]", "", regex=True)
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
        final_url = resolve_url(submit_url, base_url) if base_url else resolve_url(submit_url, None)
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

# ---------- Optional LLM helpers ----------
def llm_available():
    """
    Return True if an LLM client looks configured.
    Currently supports OpenAI python package if OPENAI_API_KEY is set.
    """
    try:
        if os.environ.get("OPENAI_API_KEY"):
            import openai  # noqa: F401
            return True
    except Exception:
        pass
    return False

def llm_query(prompt, max_tokens=256, model="gpt-4o-mini" ):
    """
    Lightweight wrapper to query an LLM (OpenAI python lib).
    If LLM isn't configured, returns None.
    """
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        import openai
        # conservative parameters - caller should sanitize prompt length
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        # Try to extract the assistant text
        if resp and "choices" in resp and len(resp["choices"]) > 0:
            text = resp["choices"][0]["message"]["content"]
            return text
    except Exception as e:
        logger.debug("LLM query failed: %s", e)
    return None

def detect_llm_need(result, body_text, parsed):
    """
    Small heuristic to decide when LLM might help:
      - extracting ambiguous secrets from noisy text
      - transcribing audio / video pages (if words like 'audio' appear)
      - interpreting a textual prompt that asks for nontrivial reasoning
    Adds fields to result: llm_detection
    """
    reasons = []
    if body_text:
        # if 'secret' appears with ambiguous surrounding text
        if re.search(r"secret|secret code|the secret code|code is", body_text, flags=re.IGNORECASE):
            reasons.append("extract_secret_possible")
        # audio / transcription hint
        if re.search(r"\baudio\b|\btranscribe\b|\bwav\b|\bmp3\b|\brecording\b", body_text, flags=re.IGNORECASE):
            reasons.append("transcription_audio")
        # if there is a CSV instruction but with fuzzy description
        if re.search(r"csv|cutoff|sum|filter", body_text, flags=re.IGNORECASE) and not parsed:
            reasons.append("csv_processing_helpful")
    need = len(reasons) > 0
    result["llm_detection"] = {"need": need, "why": reasons}
    return need, reasons

# ---------- Main solver ----------
def solve_quiz(email, secret, url, deadline):
    """
    Visit the URL, attempt to extract the task, compute an answer, and post it to the
    submit URL found on the page. Returns a result dict (with debug fields).
    """
    if time.time() > deadline:
        raise TimeoutError("Deadline already passed")

    result = {"start_url": url, "ts": time.time()}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # navigate
        page.goto(url, wait_until="networkidle", timeout=30000)
        time.sleep(0.05)

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

        # detect whether an LLM might help
        try:
            need_llm, llm_reasons = detect_llm_need(result, body_text, parsed)
            result["llm"] = {"enabled": llm_available()}
            result["llm_detection"] = result.get("llm_detection", {})
            if need_llm and llm_available():
                # quick LLM attempt (non-blocking errors)
                prompt = "Extract the most likely secret code or numeric token from this text:\n\n" + (body_text or "")
                try:
                    q = llm_query(prompt, max_tokens=256)
                    result["llm_quick_response"] = q
                except Exception as e:
                    result["llm_quick_response_error"] = str(e)
        except Exception as e:
            logger.debug("LLM detection error: %s", e)

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
                        time.sleep(0.05)
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


        try:
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
                    # collect likely CSV links (file extension or text hint)
                    if href_l.endswith(".csv") or "csv" in (type_attr or "").lower() or ".csv?" in href_l:
                        csv_links.append(href)
                    elif ".csv" in (text or "").lower():
                        csv_links.append(href)
            except Exception:
                csv_links = []

            # also try to pick up CSV-like relative links from body text as fallback
            if not csv_links and body_text:
                m_csv_rel = re.search(r"(\/[^\s'\"<>]*?(?:csv|demo-audio-data)[^\s'\"<>]*)", body_text, flags=re.IGNORECASE)
                if m_csv_rel:
                    csv_links.append(m_csv_rel.group(1))

            if csv_links:
                chosen = csv_links[0]
                csv_url = resolve_url(chosen, page.url)
                result["detected_csv_links"] = csv_links
                result["resolved_csv_url"] = csv_url

                # detect cutoff if present on the page
                cutoff = None
                if body_text:
                    m_cut = re.search(r"Cutoff[:\s]*([0-9]+(?:\.[0-9]+)?)", body_text, flags=re.IGNORECASE)
                    if m_cut:
                        cutoff = float(m_cut.group(1))
                        result["detected_cutoff"] = cutoff

                # download the CSV
                try:
                    r = httpx.get(csv_url, timeout=30.0)
                    r.raise_for_status()
                    tmpdir = tempfile.mkdtemp()
                    csv_path = os.path.join(tmpdir, "data.csv")
                    with open(csv_path, "wb") as f:
                        f.write(r.content)
                    result["downloaded_csv"] = csv_path

                    # Try reading CSV robustly:
                    # 1) Try with header=0 (default). If first row looks numeric-only -> treat file as headerless.
                    # 2) If headerless, read with header=None and name column 'value' or use integer column index.
                    read_attempts = []
                    df = None
                    # attempt 1: default read (pandas will infer header)
                    try:
                        df_try = pd.read_csv(csv_path)
                        read_attempts.append(("header_inferred", df_try.shape, list(df_try.columns[:5]) if df_try.shape[1] <= 10 else list(df_try.columns[:5]) + ["..."]))
                        # heuristic: if the column names are numeric strings (e.g., '96903') or the first column name looks like a data value,
                        # treat as headerless.
                        first_col_name = str(df_try.columns[0])
                        if re.fullmatch(r"\d+(\.\d+)?", first_col_name) or all(re.fullmatch(r"-?\d+(\.\d+)?", str(x)) for x in df_try.columns[:1]):
                            # probably header-less CSV where pandas used first row as header
                            read_attempts.append(("header_inferred_but_numeric_colname", first_col_name))
                            df = pd.read_csv(csv_path, header=None)
                        else:
                            df = df_try
                    except Exception as e:
                        read_attempts.append(("read_error", str(e)))
                        try:
                            df = pd.read_csv(csv_path, header=None)
                            read_attempts.append(("read_header_none", df.shape))
                        except Exception as e2:
                            read_attempts.append(("read_header_none_error", str(e2)))
                            df = None

                    result["csv_read_attempts"] = read_attempts

                    if df is None:
                        result["csv_read_error"] = "unable to parse csv"
                    else:
                        # If header=None, give a sensible column name
                        if df.columns.dtype == "int64" or any(isinstance(c, int) for c in df.columns):
                            # single column without header -> name it 'value'
                            if df.shape[1] == 1:
                                df.columns = ["value"]
                            else:
                                # multiple unnamed columns -> keep numeric column names as-is (0,1,2..)
                                df.columns = [f"col_{i}" for i in range(df.shape[1])]

                        # If the CSV contains a single column with header that looks numeric (we renamed above), pick that.
                        numeric_cols = [c for c in df.columns if is_numeric_series(df[c])]
                        result["csv_shape"] = df.shape
                        result["csv_numeric_columns"] = numeric_cols

                        # If no numeric columns detected, try coercing the first column
                        if not numeric_cols and df.shape[1] >= 1:
                            # coerce first column as fallback
                            df.iloc[:, 0] = df.iloc[:, 0].astype(str)
                            df.iloc[:, 0] = df.iloc[:, 0].str.replace(r"[, ]", "", regex=True)  # remove thousands separators/spaces
                            df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
                            numeric_cols = [df.columns[0]]
                            result["csv_numeric_columns"] = numeric_cols

                        # Clean numeric column and compute sum according to cutoff
                        first_col = numeric_cols[0]
                        # clean common non-numeric artifacts
                        df[first_col] = df[first_col].astype(str).str.replace(r"[^\d\.\-\+eE]", "", regex=True)
                        df[first_col] = pd.to_numeric(df[first_col], errors="coerce")
                        # drop NaNs
                        numeric_series = df[first_col].dropna()

                        # Heuristic for cutoff direction:
                        # If most values are > cutoff, maybe intent is sum values >= cutoff (existing). If most are < cutoff maybe they meant < cutoff.
                        answer_value = None
                        if cutoff is not None:
                            if numeric_series.empty:
                                answer_value = 0.0
                            else:
                                above_pct = (numeric_series >= cutoff).mean()
                                below_pct = (numeric_series <= cutoff).mean()
                                # prefer >= cutoff unless majority below cutoff
                                if above_pct >= 0.5:
                                    mask = numeric_series >= cutoff
                                else:
                                    mask = numeric_series <= cutoff
                                answer_value = float(numeric_series[mask].sum(min_count=1) or 0.0)
                                result["csv_cutoff_direction_used"] = ">= cutoff" if above_pct >= 0.5 else "<= cutoff"
                        else:
                            answer_value = float(numeric_series.sum(min_count=1) or 0.0)
                            result["csv_cutoff_direction_used"] = "no_cutoff_sum_all"

                        result["csv_answer"] = answer_value

                        # Prepare payload (use original start_url, not CSV helper)
                        submit_payload = {"email": email, "secret": secret, "url": result.get("start_url") or url, "answer": answer_value}

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
                    result["csv_download_error"] = str(e)
        except Exception as e:
            logger.debug("csv-detection error: %s", e)

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
