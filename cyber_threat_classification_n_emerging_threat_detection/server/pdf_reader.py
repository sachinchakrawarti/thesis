"""Extract text from PDF bytes using PyMuPDF."""
import fitz

def extract_text_from_bytes(data: bytes):
    if not data or not data.startswith(b"%PDF"):
        return "", "not a valid PDF"
    doc = None
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        if doc.page_count == 0:
            return "", "0 pages"
        return "\n".join(page.get_text() for page in doc), None
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"
    finally:
        if doc:
            try:
                doc.close()
            except Exception:
                pass
