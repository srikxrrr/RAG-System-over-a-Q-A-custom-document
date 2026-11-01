"""Clause-aware chunking.

Engineering documents are organised by numbered clauses ("4.2 Nominal cover"), tables and
annexes. Splitting every N tokens destroys that structure, so we:
  1. detect headings / clause numbers and track the section path,
  2. keep each table together (with its caption, header row repeated if it must be split),
  3. pack paragraphs inside one clause up to `max_words`, never crossing a clause boundary,
  4. attach clause number, section path and page to every chunk for exact citations.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .loaders import Page

HEAD_MD = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
HEAD_NUM = re.compile(r"^(\d+(?:\.\d+){0,4})\.?\s+([A-Z][^\n]{1,100})$")
HEAD_ANNEX = re.compile(r"^((?:Annex|Appendix)\s+[A-Z0-9]+)\b[\s:.\-–—]*(.*)$", re.I)
CLAUSE_PREFIX = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$")
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
TABLE_CAPTION = re.compile(r"^\W*Table\s+\d+(?:\.\d+)*", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    source: str
    page: int | None
    clause: str          # "4.2" or "" if the document has no numbering
    heading: str         # "Nominal cover"
    section_path: str    # "Durability > Nominal cover"
    ordinal: int         # position inside the document
    text: str

    @property
    def index_text(self) -> str:
        """What gets embedded / BM25-indexed: body plus its context header."""
        head = f"{self.title} | {self.section_path}"
        if self.clause:
            head += f" | clause {self.clause}"
        return f"{head}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def _words(s: str) -> int:
    return len(s.split())


def _heading(line: str):
    """Return (level, clause, label) if `line` is a heading, else None."""
    s = line.strip()
    m = HEAD_MD.match(s)
    if m:
        level, text = len(m.group(1)), m.group(2).strip("# ").strip()
        cm = CLAUSE_PREFIX.match(text)
        return (level, cm.group(1), cm.group(2)) if cm else (level, "", text)
    m = HEAD_ANNEX.match(s)
    if m and _words(s) <= 12 and not s.endswith("."):
        label = m.group(1) + (f" {m.group(2)}" if m.group(2) else "")
        return (1, m.group(1), label)
    m = HEAD_NUM.match(s)
    if m and _words(s) <= 12 and not s.endswith("."):
        return (m.group(1).count(".") + 1, m.group(1), m.group(2))
    return None


@dataclass
class _Section:
    clause: str
    heading: str
    path: str
    lines: list  # [(page, text)]


def _split_sections(pages: list[Page]) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    cur = _Section("", "", "", [])
    doc_title = pages[0].title if pages else ""
    for pg in pages:
        for line in pg.text.splitlines():
            h = _heading(line)
            if h:
                level, clause, label = h
                if cur.lines or cur.heading:
                    sections.append(cur)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, label))
                path_parts = [lab for _, lab in stack if lab != doc_title]
                cur = _Section(clause, label, " > ".join(path_parts), [])
            else:
                cur.lines.append((pg.page, line))
    if cur.lines or cur.heading:
        sections.append(cur)
    return sections


def _blocks(lines: list) -> list[tuple[str, int | None, str]]:
    """Group lines into ('para'|'table', first_page, text) blocks; glue table captions to tables."""
    blocks: list[list] = []
    buf: list[str] = []
    kind, first_page = None, None

    def flush():
        nonlocal buf, kind, first_page
        if buf:
            blocks.append([kind, first_page, "\n".join(buf).strip()])
        buf, kind, first_page = [], None, None

    for pg, line in lines:
        if not line.strip():
            flush()
            continue
        k = "table" if TABLE_ROW.match(line) else "para"
        if kind and k != kind:
            flush()
        if not buf:
            kind, first_page = k, pg
        buf.append(line.rstrip())
    flush()
    merged: list[list] = []
    for b in blocks:
        if (b[0] == "table" and merged and merged[-1][0] == "para"
                and TABLE_CAPTION.match(merged[-1][2]) and _words(merged[-1][2]) <= 25):
            cap = merged.pop()
            b = ["table", cap[1], cap[2] + "\n" + b[2]]
        merged.append(b)
    return [(k, p, t) for k, p, t in merged if t]


def _split_para(text: str, max_words: int) -> list[str]:
    out, cur = [], []
    for sent in SENT_SPLIT.split(text.replace("\n", " ")):
        if cur and _words(" ".join(cur + [sent])) > max_words:
            out.append(" ".join(cur))
            cur = []
        cur.append(sent)
    if cur:
        out.append(" ".join(cur))
    return out


def _split_table(text: str, max_words: int) -> list[str]:
    lines = text.splitlines()
    caption = [ln for ln in lines if not TABLE_ROW.match(ln)]
    rows = [ln for ln in lines if TABLE_ROW.match(ln)]
    header = rows[:2] if len(rows) > 1 and TABLE_SEP.match(rows[1]) else rows[:1]
    body = rows[len(header):]
    out, cur = [], []
    for r in body:
        if cur and _words("\n".join(caption + header + cur + [r])) > max_words:
            out.append("\n".join(caption + header + cur))
            cur = []
        cur.append(r)
    if cur:
        out.append("\n".join(caption + header + cur))
    return out or [text]


def chunk_document(pages: list[Page], max_words: int = 220, overlap_words: int = 30) -> list[Chunk]:
    if not pages:
        return []
    first = pages[0]
    chunks: list[Chunk] = []

    for sec in _split_sections(pages):
        pieces: list[tuple[str, int | None, str]] = []  # (kind, page, text) already <= max_words
        for kind, pg, text in _blocks(sec.lines):
            if _words(text) <= max_words:
                pieces.append((kind, pg, text))
            elif kind == "table":
                pieces += [("table", pg, t) for t in _split_table(text, max_words)]
            else:
                pieces += [("para", pg, t) for t in _split_para(text, max_words)]

        cur: list[str] = []
        cur_page: int | None = None
        n = 0
        last_para_tail = ""

        def emit():
            nonlocal cur, cur_page, n
            if not cur:
                return
            chunks.append(Chunk(
                id=f"{first.doc_id}::{len(chunks):04d}", doc_id=first.doc_id, title=first.title,
                source=first.source, page=cur_page, clause=sec.clause, heading=sec.heading,
                section_path=sec.path, ordinal=len(chunks), text="\n\n".join(cur).strip()))
            cur, cur_page, n = [], None, 0

        for kind, pg, text in pieces:
            w = _words(text)
            if cur and n + w > max_words:
                emit()
                if overlap_words and last_para_tail:
                    cur, n = [last_para_tail], _words(last_para_tail)
                    cur_page = pg
            if not cur or cur_page is None:
                cur_page = pg if cur_page is None else cur_page
            cur.append(text)
            n += w
            if kind == "para":
                last_para_tail = " ".join(text.split()[-overlap_words:]) if overlap_words else ""
            else:
                last_para_tail = ""
        emit()
    return chunks
