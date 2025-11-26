# app/utils/pdf_utils.py
# Minimal helper to extract tables using pdfplumber into pandas DataFrames.
import pdfplumber
import pandas as pd
import re

def extract_tables_from_pdf(pdf_path):
    """
    Return dict mapping 1-based page numbers -> list of DataFrames found on that page.
    """
    tables_by_page = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables()
                    dfs = []
                    for t in tables:
                        # create DataFrame; handle ragged rows
                        df = pd.DataFrame(t[1:], columns=t[0]) if len(t) > 1 else pd.DataFrame()
                        dfs.append(df)
                    if dfs:
                        tables_by_page[i] = dfs
                except Exception:
                    continue
    except Exception:
        pass
    return tables_by_page
