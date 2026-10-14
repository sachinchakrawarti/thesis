# scripts/document_extractor.py
"""
Extract plain text from uploaded CTI documents (PDF, DOCX, TXT, MD, HTML).

Requires:
    pip install pypdf python-docx beautifulsoup4 lxml
"""

import os
import re
from pathlib import Path
from typing import Union


# ------------------------------------------------------------------
# Individual extractors
# ------------------------------------------------------------------
def _extract_txt(path: Path) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # legacy fallback
        except ImportError:
            raise ImportError(
                "Install pypdf:  pip install pypdf"
            )
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def _extract_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError:
        raise ImportError(
            "Install python-docx:  pip install python-docx"
        )
    doc = docx.Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text for c in row.cells))
    return "\n".join(parts)


def _extract_html(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "Install beautifulsoup4:  pip install beautifulsoup4 lxml"
        )
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n")


# ------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------
EXTRACTORS = {
    ".txt": _extract_txt,
    ".md":  _extract_txt,
    ".log": _extract_txt,
    ".csv": _extract_txt,
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".doc": _extract_docx,   # may fail for legacy .doc
    ".html": _extract_html,
    ".htm": _extract_html,
}


def extract_text(path: Union[str, Path]) -> str:
    """Extract text from a supported document. Raises on failure."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in EXTRACTORS:
        raise ValueError(
            f"Unsupported extension: {ext}. "
            f"Supported: {', '.join(sorted(EXTRACTORS))}"
        )
    return EXTRACTORS[ext](path)


def clean_extracted_text(text: str) -> str:
    """Normalise whitespace, drop page artefacts, strip null chars."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    # remove repeated page numbers / headers like "Page 3 of 12"
    text = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", " ", text, flags=re.I)
    # collapse hyphenated line breaks: "threat-\nintel" -> "threatintel"
    text = re.sub(r"-\n", "", text)
    # join wrapped lines
    text = re.sub(r"\n{2,}", "\n\n", text)
    # collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()


def extract_and_clean(path: Union[str, Path]) -> str:
    return clean_extracted_text(extract_text(path))


# ------------------------------------------------------------------
# CLI usage:  python document_extractor.py path\to\file.pdf
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python document_extractor.py <file>")
        sys.exit(1)
    txt = extract_and_clean(sys.argv[1])
    print(txt[:2000])