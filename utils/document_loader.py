"""
Document loader for the RAG knowledge assistant.

Routes an uploaded file to the correct extractor based on its extension
and returns a list of {"text", "metadata"} records. Metadata carries the
source filename (and page number for PDFs) so answers can cite sources.

Supported: .pdf .docx .pptx .xlsx .xls .csv .tsv .rtf .html .htm .md .txt .json
Images and the legacy binary .doc format are not supported.
"""

import os
import json

import pandas as pd
from pypdf import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
from striprtf.striprtf import rtf_to_text
from bs4 import BeautifulSoup
import markdown as md_lib


# Extensions this loader can read
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls",
    ".csv", ".tsv", ".rtf", ".html", ".htm",
    ".md", ".txt", ".json",
}


def _clean(text):
    """Collapse excess whitespace and strip blank lines."""
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


# ---------- Per-format extractors ----------

def _load_pdf(path, source):
    reader = PdfReader(path)
    records = []
    for i, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            records.append({"text": text, "metadata": {"source": source, "page": i}})
    return records


def _load_docx(path, source):
    doc = DocxDocument(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    # Include table cell text
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = _clean("\n".join(parts))
    return [{"text": text, "metadata": {"source": source}}] if text else []


def _load_pptx(path, source):
    prs = Presentation(path)
    records = []
    for i, slide in enumerate(prs.slides, start=1):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in para.runs).strip()
                    if line:
                        parts.append(line)
        text = _clean("\n".join(parts))
        if text:
            records.append({"text": text, "metadata": {"source": source, "slide": i}})
    return records


def _load_spreadsheet(path, source, sep=None):
    ext = os.path.splitext(path)[1].lower()
    records = []
    if ext in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)  # all sheets
        for name, df in sheets.items():
            text = _clean(df.to_csv(index=False))
            if text:
                records.append({"text": text, "metadata": {"source": source, "sheet": name}})
    else:
        delimiter = "\t" if ext == ".tsv" else (sep or ",")
        df = pd.read_csv(path, sep=delimiter)
        text = _clean(df.to_csv(index=False))
        if text:
            records.append({"text": text, "metadata": {"source": source}})
    return records


def _load_rtf(path, source):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = _clean(rtf_to_text(f.read()))
    return [{"text": text, "metadata": {"source": source}}] if text else []


def _load_html(path, source):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = _clean(soup.get_text(separator="\n"))
    return [{"text": text, "metadata": {"source": source}}] if text else []


def _load_markdown(path, source):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = md_lib.markdown(f.read())
    text = _clean(BeautifulSoup(html, "lxml").get_text(separator="\n"))
    return [{"text": text, "metadata": {"source": source}}] if text else []


def _load_txt(path, source):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = _clean(f.read())
    return [{"text": text, "metadata": {"source": source}}] if text else []


def _load_json(path, source):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    text = _clean(json.dumps(data, indent=2, ensure_ascii=False))
    return [{"text": text, "metadata": {"source": source}}] if text else []


# ---------- Dispatcher ----------

_LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".pptx": _load_pptx,
    ".xlsx": _load_spreadsheet,
    ".xls": _load_spreadsheet,
    ".csv": _load_spreadsheet,
    ".tsv": _load_spreadsheet,
    ".rtf": _load_rtf,
    ".html": _load_html,
    ".htm": _load_html,
    ".md": _load_markdown,
    ".txt": _load_txt,
    ".json": _load_json,
}


def load_document(path):
    """
    Extract text from a single file.

    Args:
        path: Path to the file on disk.

    Returns:
        List of {"text": str, "metadata": dict} records. Empty list if the
        file has no readable text.

    Raises:
        ValueError: If the file type is not supported.
    """
    source = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    return loader(path, source)


def load_documents(paths):
    """
    Extract text from many files. Files that fail are skipped and reported.

    Args:
        paths: Iterable of file paths.

    Returns:
        (records, errors)
          records: combined list of {"text", "metadata"} from all files
          errors:  list of {"source", "error"} for files that failed
    """
    records, errors = [], []
    for path in paths:
        source = os.path.basename(path)
        try:
            records.extend(load_document(path))
        except Exception as e:
            errors.append({"source": source, "error": str(e)})
    return records, errors
