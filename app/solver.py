# app/solver.py
import time
import json
import re
import os
import tempfile
import logging
import httpx
import pandas as pd
from playwright.sync_api import sync_playwright
from .utils.pdf_utils import extract_tables_from_pdf

logger = logging.getLogger(__name__)

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
    except Exception as e:
        logger.debug("find_submit_url error: %s", e)
    return None

def is_numeric_series(series, threshold=0.6):
    s = series.astype(str).str.replace(r"[^0-9.-]", "", regex=True)
    parsed = pd.to_numeric(s, errors="coerce")
    non_null = parsed.notna().sum()
    total = len(parsed)
    if total == 0:
        return False
    return (non_null / total) >= threshold

def try_submit_json(submit_url, payload, timeout=30.0):
    try:
        r = httpx.post(submit_url, json=payload, timeout=timeout)
        try:
            body = r.json()
        except Exception:
            body = r.text
        return {"status_code": r.status_code, "body": body}
    except Exception as e:
        return {"error": str(e)}

def solve_quiz(email, secret, url, deadline):
    if time.time() > deadline:
        raise TimeoutError("Deadline already passed")

    result = {"start_url": url, "ts": time.time()}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        page.goto(url, wait_until="networkidle", timeout=30000)
        time.sleep(0.2)

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

        submit_url = find_submit_url_from_page(page)
        result["submit_url"] = submit_url

        if not submit_url and body_text:
            m = re.search(r"(https?://[^\s'\"<>]*?/submit[^\s'\"<>]*)", body_text, flags=re.IGNORECASE)
            if m:
                submit_url = m.group(1)
                result["submit_url"] = submit_url
                result["submit_url_extracted_from_text"] = True

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
            else:
                parsed = None
        except Exception:
            parsed = None

        # Detect PDF link
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
                with page.expect_download() as dl:
                    try:
                        page.click(f'a[href="{pdf_link}"]')
                    except Exception:
                        page.goto(pdf_link)
                download = dl.value
                pdf_name = download.suggested_filename or "download.pdf"
                pdf_path = os.path.join(tmpdir, pdf_name)
                download.save_as(pdf_path)
                result["downloaded_pdf"] = pdf_path

                tables_by_page = extract_tables_from_pdf(pdf_path)
                result["pdf_tables_count"] = sum(len(dfs) for dfs in tables_by_page.values())

                answer_payload = None
                if 2 in tables_by_page:
                    for df in tables_by_page[2]:
                        cols_lower = [str(c).lower() for c in df.columns]
                        if "value" in cols_lower:
                            colname = df.columns[cols_lower.index("value")]
                            df[colname] = df[colname].astype(str).str.replace(r"[^0-9.-]", "", regex=True)
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

                if not answer_payload:
                    for page_no, dfs in tables_by_page.items():
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
                                result["answer_reason"] = f"sum_first_numeric_col_page_{page_no}"
                                result["answer"] = total
                                break
                        if answer_payload:
                            break

                if not answer_payload:
                    import pdfplumber
                    with pdfplumber.open(pdf_path) as pdf:
                        if len(pdf.pages) >= 2:
                            text2 = pdf.pages[1].extract_text() or ""
                            nums = re.findall(r"[-]?[0-9,]+(?:\.[0-9]+)?", text2)
                            nums_clean = [float(n.replace(",", "")) for n in nums] if nums else []
                            if nums_clean:
                                answer_payload = {
                                    "email": email,
                                    "secret": secret,
                                    "url": url,
                                    "answer": sum(nums_clean)
                                }
                                result["answer_reason"] = "sum_numbers_from_text_page_2"
                                result["answer"] = answer_payload["answer"]

                if answer_payload and submit_url:
                    submit_result = try_submit_json(submit_url, answer_payload)
                    result["submit_result"] = submit_result

                return result

            except Exception as e:
                logger.exception("PDF processing error")
                result["pdf_error"] = str(e)
                return result

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
            if answer_payload and submit_url:
                submit_result = try_submit_json(submit_url, answer_payload)
                result["submit_result"] = submit_result
            return result
        except Exception:
            pass

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
                    submit_result = try_submit_json(submit_url, parsed)
                    result["submit_result"] = submit_result
                    return result
                else:
                    result["pre_parsed_but_no_submit_url"] = True
            except Exception as e:
                result["pre_submit_error"] = str(e)

        try:
            if body_text:
                m2 = re.search(r"(https?://[^\s'\"<>]*?(submit|answer)[^\s'\"<>]*)", body_text, flags=re.IGNORECASE)
                if m2:
                    candidate = m2.group(1)
                    result["fallback_submit_candidate"] = candidate
                    if parsed and isinstance(parsed, dict):
                        submit_result = try_submit_json(candidate, parsed)
                        result["submit_result"] = submit_result
                        return result
        except Exception:
            pass

        return result
