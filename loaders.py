"""Turn files (PDF / DOCX / MD / TXT) into clean text 'pages' with provenance."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    doc_id: str
    title: str
    source: str          # file name shown in citations
    page: int | None     # 1-based for PDFs, None otherwise
    text: str


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _strip_front_matter(text: str) -> tuple[str, str | None]:
    """Remove a leading YAML block; return (body, title-if-present)."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            block = text[3:end]
            m = re.search(r"^title:\s*(.+)$", block, re.M)
            return text[end + 4:].lstrip("\n"), (m.group(1).strip().strip("\"'") if m else None)
    return text, None


def _md_title(text: str) -> str | None:
    m = re.search(r"^#\s+(.+?)\s*$", text, re.M)
    return m.group(1) if m else None


def _clean_pdf_pages(pages: list[str]) -> list[str]:
    """Drop running headers/footers and bare page numbers; re-join hyphenated line breaks."""
    common: set[str] = set()
    if len(pages) >= 4:
        counts: Counter = Counter()
        for p in pages:
            lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
            for ln in set(lines[:2] + lines[-2:]):
                counts[re.sub(r"\d+", "#", ln)] += 1
        common = {k for k, c in counts.items() if c >= 0.6 * len(pages)}
    out = []
    for p in pages:
        keep = []
        for ln in p.splitlines():
            s = ln.strip()
            if s and re.sub(r"\d+", "#", s) in common:
                continue
            if re.fullmatch(r"(page\s*)?\d+(\s*of\s*\d+)?", s, re.I):
                continue
            keep.append(ln)
        t = "\n".join(keep)
        t = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", t)
        out.append(t)
    return out


def _load_pdf(path: Path, doc_id: str) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    raw = [(pg.extract_text() or "") for pg in reader.pages]
    cleaned = _clean_pdf_pages(raw)
    meta_title = (reader.metadata.title if reader.metadata else None) or None
    title = meta_title or path.stem.replace("_", " ")
    pages = [Page(doc_id, title, path.name, i + 1, t) for i, t in enumerate(cleaned) if t.strip()]
    if not pages:
        print(f"  ! {path.name}: no text layer found (scanned PDF?). OCR it first, e.g. with ocrmypdf.")
    return pages


def _load_docx(path: Path, doc_id: str) -> list[Page]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = Document(str(path))
    lines: list[str] = []
    title = None
    for child in d.element.body.iterchildren():
        if child.tag.endswith("}p"):
            p = Paragraph(child, d)
            txt = p.text.strip()
            if not txt:
                lines.append("")
                continue
            style = (p.style.name or "") if p.style is not None else ""
            m = re.match(r"Heading (\d)", style)
            if style == "Title" and not title:
                title = txt
            lines.append(("#" * int(m.group(1)) + " " + txt) if m else txt)
        elif child.tag.endswith("}tbl"):
            t = Table(child, d)
            rows = [" | ".join(c.text.strip().replace("\n", " ") for c in r.cells) for r in t.rows]
            if rows:
                lines.append("")
                lines.append("| " + rows[0] + " |")
                lines.append("|" + "---|" * len(t.rows[0].cells))
                lines.extend("| " + r + " |" for r in rows[1:])
                lines.append("")
    return [Page(doc_id, title or path.stem.replace("_", " "), path.name, None, "\n".join(lines))]


def load_file(path: Path) -> list[Page]:
    ext = path.suffix.lower()
    doc_id = slug(path.stem)
    if ext in (".md", ".txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        body, fm_title = _strip_front_matter(text)
        title = fm_title or _md_title(body) or path.stem.replace("_", " ")
        return [Page(doc_id, title, path.name, None, body)]
    if ext == ".pdf":
        return _load_pdf(path, doc_id)
    if ext == ".docx":
        return _load_docx(path, doc_id)
    raise ValueError(f"Unsupported file type: {ext}")


def discover(folder: Path, extensions: list[str]) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
