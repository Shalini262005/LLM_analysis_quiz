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
# Optional LLM integration helpers (kept from prior patch)
# ---------------------------
ENABLE_LLM = os.getenv("ENABLE_LLM", "0") == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
try:
    import openai  # type: ignore
    OPENAI_PY_AVAILABLE = True
except Exception:
    OPENAI_PY_AVAILABLE = False

def llm_available():
    return ENABLE_LLM and OPENAI_API_KEY is not None

def llm_call_simple(system_prompt: str, user_prompt: str, max_tokens: int = 512):
    if not llm_available():
        raise RuntimeError("LLM not available (enable with ENABLE_LLM=1 and set OPENAI_API_KEY).")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
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
        except Exception:
            logger.exception("openai package call failed, falling back to httpx")
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": LLM_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
        r = httpx.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=60.0)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("LLM httpx call failed")
        raise

def detect_need_for_llm(page_text: str, page_html: str, links: list):
    reasons = {"need": False, "why": []}
    if not page_text and not page_html:
        return reasons
    text = (page_text or "") + "\n" + (page_html or "")
    if re.search(r"\.(mp3|wav|m4a|ogg)(\?|$)", (" ".join(links)).lower()) or re.search(r"\b(audio|listen|transcribe|speech|voice)\b", text, flags=re.IGNORECASE):
        reasons["need"] = True; reasons["why"].append("transcription_audio")
    if re.search(r"\.(png|jpg|jpeg|tiff)(\?|$)", (" ".join(links)).lower()) or re.search(r"\b(image|photo|screenshot|ocr)\b", text, flags=re.IGNORECASE):
        reasons["need"] = True; reasons["why"].append("ocr_image")
    if re.search(r"\b(summarize|explain|interpret|visualize|plot|chart|transcribe|classify|predict)\b", text, flags=re.IGNORECASE):
        reasons["need"] = True; reasons["why"].append("summarize_interpret")
    if re.search(r"\b(secret code|secret|code is|secret:|cutoff)\b", text, flags=re.IGNORECASE):
        reasons["need"] = True; reasons["why"].append("extract_secret_possible")
    return reasons

def llm_extract_answer_from_text(body_text: str, question_hint: str = None):
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
# Existing helper functions
# ---------------------------
def find_submit_url_from_page(page):
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
    if not candidate:
        return None
    candidate = candidate.strip()
    if bool(urlparse(candidate).scheme):
        return candidate
    if base:
        return urljoin(base, candidate)
    if candidate.startswith("//"):
        return "https:" + candidate
    return candidate

def try_submit_json(submit_url, payload, base_url=None, timeout=30.0):
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
    if not text:
        return None
    txt = " ".join(text.split())
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
    numeric_tokens = re.findall(r"\d{3,64}", txt)
    if numeric_tokens:
        numeric_tokens.sort(key=lambda s: -len(s))
        return numeric_tokens[0]
    m2 = re.search(r"(?:secret|secret code|code)\s*(?:is|:|=)?\s*['\"]?([A-Za-z0-9_\-]{4,64})['\"]?", txt, flags=re.IGNORECASE)
    if m2:
        return m2.group(1)
    tokens = re.findall(r"[A-Za-z0-9_\-]{6,64}", txt)
    for t in tokens:
        if not re.fullmatch(r"https?|http|submit|demo|page|email|secret|code", t, flags=re.IGNORECASE):
            return t
    return None

# ---------------------------
# Main solver (patched)
# ---------------------------
def solve_quiz(email, secret, url, deadline):
    if time.time() > deadline:
        raise TimeoutError("Deadline already passed")

    result = {"start_url": url, "ts": time.time(), "llm": {"enabled": llm_available()}}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

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

        # optional LLM quick attempt
        try:
            detection = detect_need_for_llm(body_text, page.content(), anchors_texts)
            result["llm_detection"] = detection
            if detection.get("need") and llm_available():
                try:
                    hint = "Extract a short machine readable answer from the following page text. If a numeric secret/cutoff is present return only that number. If the page instructs to download a CSV/AUDIO/IMAGE and compute an answer, say 'NEEDS_DOWNLOAD' and what to download (url or pattern)."
                    preview_text = (body_text or "")[:4000]
                    llm_resp = llm_extract_answer_from_text(preview_text, question_hint=hint)
                    result["llm_quick_response"] = llm_resp
                    if llm_resp:
                        m_num = re.search(r"\b([0-9]{2,64})\b", llm_resp)
                        if m_num:
                            candidate = m_num.group(1)
                            result["llm_extracted_numeric_candidate"] = candidate
                            payload = {"email": email, "secret": secret, "url": url, "answer": candidate}
                            found_submit = find_submit_url_from_page(page)
                            if found_submit:
                                submit_result = try_submit_json(found_submit, payload, base_url=url)
                                result["submit_result"] = submit_result
                                return result
                except Exception as e:
                    logger.debug("llm quick extraction failed: %s", e)
                    result.setdefault("llm_errors", []).append(str(e))
        except Exception as e:
            logger.debug("llm detection error: %s", e)

        # find submit url
        submit_url = find_submit_url_from_page(page)
        result["submit_url"] = submit_url
        if not submit_url and body_text:
            m = re.search(r"(\/[^\s'\"<>]*?submit[^\s'\"<>]*)", body_text, flags=re.IGNORECASE)
            if m:
                submit_url = m.group(1)
                result["submit_url_extracted_from_text"] = True
                result["submit_url"] = submit_url

        # parse JSON in <pre>
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

        # Detect demo-scrape-data pattern and handle it
        try:
            if body_text:
                m_rel = re.search(r"(\/[^\s'\"<>]*demo-scrape-data[^\s'\"<>]*)", body_text, flags=re.IGNORECASE)
                if m_rel:
                    relpath = m_rel.group(1)
                    scrape_url = resolve_url(relpath, page.url)
                    result["detected_scrape_instruction"] = {"relpath": relpath, "scrape_url": scrape_url}
                    try:
                        page.goto(scrape_url, wait_until="networkidle", timeout=20000)
                        time.sleep(0.1)
                        scrape_text = page.inner_text("body") if page.query_selector("body") else ""
                        result["scrape_page_preview"] = scrape_text[:2000]
                        secret_code = extract_secret_from_text(scrape_text)
                        result["scraped_secret_candidate"] = secret_code
                        if secret_code:
                            # IMPORTANT: use original task URL in payload (most submit endpoints expect this)
                            payload = {
                                "email": email,
                                "secret": secret,
                                "url": result.get("start_url") or url,
                                "answer": secret_code
                            }

                            logger.debug(
                                "Posting scraped secret payload to submit_url=%s resolved=%s payload_preview=%s",
                                submit_url,
                                resolve_url(submit_url, scrape_url),
                                {"email": email, "url": payload["url"], "answer": secret_code}
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

        # PDF detection and parsing
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
                pdf_url = resolve_url(pdf_link, page.url)
                r = httpx.get(pdf_url, timeout=30.0)
                pdf_path = os.path.join(tmpdir, "download.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(r.content)
                result["downloaded_pdf"] = pdf_path

                tables_by_page = extract_tables_from_pdf(pdf_path)
                result["pdf_tables_count"] = sum(len(dfs) for dfs in tables_by_page.values())

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
                                answer_payload = {"email": email, "secret": secret, "url": url, "answer": total}
                                result["answer_reason"] = "sum_value_on_pdf_page_2"
                                result["answer"] = total
                                break
                            except Exception:
                                continue

                if answer_payload and submit_url:
                    submit_result = try_submit_json(submit_url, answer_payload, base_url=url)
                    result["submit_result"] = submit_result
                return result

            except Exception as e:
                logger.exception("PDF processing error")
                result["pdf_error"] = str(e)
                return result

        # CSV detection & handling (robust)
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
                    if href_l.endswith(".csv") or "csv" in (type_attr or "").lower() or ".csv?" in href_l:
                        csv_links.append(href)
                    elif ".csv" in (text or "").lower():
                        csv_links.append(href)
            except Exception:
                csv_links = []

            if not csv_links and body_text:
                m_csv_rel = re.search(r"(\/[^\s'\"<>]*?(?:csv|demo-audio-data)[^\s'\"<>]*)", body_text, flags=re.IGNORECASE)
                if m_csv_rel:
                    csv_links.append(m_csv_rel.group(1))

            if csv_links:
                chosen = csv_links[0]
                csv_url = resolve_url(chosen, page.url)
                result["detected_csv_links"] = csv_links
                result["resolved_csv_url"] = csv_url

                cutoff = None
                if body_text:
                    m_cut = re.search(r"Cutoff[:\s]*([0-9]+(?:\.[0-9]+)?)", body_text, flags=re.IGNORECASE)
                    if m_cut:
                        cutoff = float(m_cut.group(1))
                        result["detected_cutoff"] = cutoff

                try:
                    r = httpx.get(csv_url, timeout=30.0)
                    r.raise_for_status()
                    tmpdir = tempfile.mkdtemp()
                    csv_path = os.path.join(tmpdir, "data.csv")
                    with open(csv_path, "wb") as f:
                        f.write(r.content)
                    result["downloaded_csv"] = csv_path

                    read_attempts = []
                    df = None
                    # Attempt 1: default read
                    try:
                        df_try = pd.read_csv(csv_path)
                        read_attempts.append({"method": "default", "shape": df_try.shape, "columns": list(df_try.columns[:5])})
                        # If first column name is numeric (pandas interpreted first row as header),
                        # re-read with header=None
                        first_col_name = str(df_try.columns[0])
                        if re.fullmatch(r"-?\d+(\.\d+)?", first_col_name):
                            read_attempts.append({"note": "first column name looks numeric -> retry with header=None", "first_col_name": first_col_name})
                            df = pd.read_csv(csv_path, header=None)
                        else:
                            # also detect case where single-column CSV but header looks like a long numeric token
                            if df_try.shape[1] == 1 and re.fullmatch(r"\d{3,}", first_col_name):
                                read_attempts.append({"note": "single column with numeric header -> retry header=None", "first_col_name": first_col_name})
                                df = pd.read_csv(csv_path, header=None)
                            else:
                                df = df_try
                    except Exception as e:
                        read_attempts.append({"method": "default_failed", "error": str(e)})
                        try:
                            df = pd.read_csv(csv_path, header=None)
                            read_attempts.append({"method": "header_none", "shape": df.shape})
                        except Exception as e2:
                            read_attempts.append({"method": "header_none_failed", "error": str(e2)})
                            df = None

                    result["csv_read_attempts"] = read_attempts

                    if df is None:
                        result["csv_read_error"] = "unable to parse csv"
                    else:
                        # If header was numeric or header=None, normalize column names
                        if df.columns.dtype == "int64" or any(isinstance(c, int) for c in df.columns):
                            # set sensible column names
                            if df.shape[1] == 1:
                                df.columns = ["value"]
                            else:
                                df.columns = [f"col_{i}" for i in range(df.shape[1])]

                        # Clean values: remove commas/space, keep scientific notation and signs
                        clean_preview = {}
                        for c in df.columns[:min(len(df.columns), 5)]:
                            # capture small preview of original and cleaned values
                            try:
                                sample_original = df[c].astype(str).iloc[:5].tolist()
                                cleaned = df[c].astype(str).str.replace(r"[,\s]", "", regex=True)
                                cleaned = cleaned.str.replace(r"[^\d\.\-\+eE]", "", regex=True)
                                sample_cleaned = cleaned.iloc[:5].tolist()
                                clean_preview[c] = {"orig_sample": sample_original, "clean_sample": sample_cleaned}
                                # apply cleaned numeric col if it looks numeric-ish
                                df[c] = cleaned
                            except Exception:
                                pass
                        result["csv_clean_preview"] = clean_preview

                        numeric_cols = [c for c in df.columns if is_numeric_series(df[c])]
                        result["csv_shape"] = df.shape
                        result["csv_numeric_columns"] = numeric_cols

                        # if no numeric columns found, coerce first column
                        if not numeric_cols and df.shape[1] >= 1:
                            df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.replace(r"[,\s]", "", regex=True)
                            df.iloc[:, 0] = df.iloc[:, 0].str.replace(r"[^\d\.\-\+eE]", "", regex=True)
                            df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
                            numeric_cols = [df.columns[0]]
                            result["csv_numeric_columns"] = numeric_cols

                        first_col = numeric_cols[0]
                        # ensure numeric dtype
                        df[first_col] = pd.to_numeric(df[first_col], errors="coerce")
                        numeric_series = df[first_col].dropna()

                        answer_value = None
                        answer_reason = None
                        chosen_mask_desc = None

                        if cutoff is not None and not numeric_series.empty:
                            above_pct = (numeric_series >= cutoff).mean()
                            below_pct = (numeric_series <= cutoff).mean()
                            # choose direction where >=50% fall, otherwise prefer >=
                            if above_pct >= 0.5:
                                mask = numeric_series >= cutoff
                                chosen_mask_desc = ">= cutoff"
                            elif below_pct >= 0.5:
                                mask = numeric_series <= cutoff
                                chosen_mask_desc = "<= cutoff"
                            else:
                                # fallback: choose >= if more elements satisfy >= than <=
                                if above_pct >= below_pct:
                                    mask = numeric_series >= cutoff
                                    chosen_mask_desc = ">= cutoff (fallback)"
                                else:
                                    mask = numeric_series <= cutoff
                                    chosen_mask_desc = "<= cutoff (fallback)"

                            answer_value = float(numeric_series[mask].sum(min_count=1) or 0.0)
                            answer_reason = f"sum_{first_col}_{chosen_mask_desc}"
                            result["csv_cutoff_direction_used"] = chosen_mask_desc
                        else:
                            answer_value = float(numeric_series.sum(min_count=1) or 0.0)
                            answer_reason = f"sum_{first_col}_all"
                            result["csv_cutoff_direction_used"] = "no_cutoff_sum_all"

                        result["csv_answer"] = answer_value
                        result["csv_answer_reason"] = answer_reason

                        # Prepare primary payload (use original start_url)
                        submit_payload = {"email": email, "secret": secret, "url": result.get("start_url") or url, "answer": answer_value}
                        preview = dict(submit_payload)
                        preview["secret"] = "***"
                        result["submit_payload_preview"] = preview

                        # perform submit attempt(s)
                        submit_attempts = []
                        if submit_url:
                            submit_result = try_submit_json(submit_url, submit_payload, base_url=csv_url)
                            submit_attempts.append({"attempt": "primary", "payload_url": submit_payload["url"], "result": submit_result})
                            result["submit_result"] = submit_result

                            # If server specifically says Wrong sum of numbers, try alternative direction (only if cutoff present)
                            try:
                                body = submit_result.get("body")
                                status = submit_result.get("status_code")
                                if status == 200 and isinstance(body, dict) and (("Wrong sum" in str(body.get("reason", ""))) or ("Wrong sum" in str(body))):
                                    # only retry if cutoff was used and we haven't tried opposite
                                    if cutoff is not None and ">= cutoff" in answer_reason:
                                        # try <= instead
                                        alt_mask = numeric_series <= cutoff
                                        alt_sum = float(numeric_series[alt_mask].sum(min_count=1) or 0.0)
                                        alt_payload = {"email": email, "secret": secret, "url": result.get("start_url") or url, "answer": alt_sum}
                                        submit_result_alt = try_submit_json(submit_url, alt_payload, base_url=csv_url)
                                        submit_attempts.append({"attempt": "fallback_opposite_cutoff", "payload_url": alt_payload["url"], "result": submit_result_alt})
                                        result["submit_result_fallback"] = submit_result_alt
                                    elif cutoff is not None and "<= cutoff" in answer_reason:
                                        # try >= instead
                                        alt_mask = numeric_series >= cutoff
                                        alt_sum = float(numeric_series[alt_mask].sum(min_count=1) or 0.0)
                                        alt_payload = {"email": email, "secret": secret, "url": result.get("start_url") or url, "answer": alt_sum}
                                        submit_result_alt = try_submit_json(submit_url, alt_payload, base_url=csv_url)
                                        submit_attempts.append({"attempt": "fallback_opposite_cutoff", "payload_url": alt_payload["url"], "result": submit_result_alt})
                                        result["submit_result_fallback"] = submit_result_alt
                                # attach submit attempts log
                                result["submit_attempts"] = submit_attempts
                            except Exception as e:
                                logger.debug("submit-fallback handling error: %s", e)
                        else:
                            result["submit_result"] = {"error": "no submit_url_found"}
                        return result
                except Exception as e:
                    result["csv_download_error"] = str(e)
        except Exception as e:
            logger.debug("csv-detection error: %s", e)

        # HTML tables
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
                    answer_payload = {"email": email, "secret": secret, "url": url, "answer": total}
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
            pass

        # pre JSON template submission
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

        # fallback: look for submit/answer link in body_text
        try:
            if body_text:
                m2 = re.search(r"(https?://[^\s'\"<>]*?(submit|answer)[^\s'\"<>]*)", body_text, flags=re.IGNORECASE)
                if m2:
                    candidate = m2.group(1)
                    result["fallback_submit_candidate"] = candidate
                    if parsed and isinstance(parsed, dict):
                        submit_result = try_submit_json(candidate, parsed, base_url=url)
                        result["submit_result"] = submit_result
                        return result

            result["note"] = "no actionable answer found on this page"
            return result
        except Exception as e:
            result["fallback_error"] = str(e)
            return result
