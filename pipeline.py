"""Orchestrates ingest and question answering."""
from __future__ import annotations

import time
from pathlib import Path

from .chunking import chunk_document
from .config import Config, load_config
from .generate import (condense_question, extractive_answer, generate_answer, llm_available)
from .index import Index
from .loaders import discover, load_file
from .retrieve import Retrieval, Retriever


def ingest(cfg: Config, data_dir: Path | None = None) -> Index:
    folder = data_dir or cfg.path(cfg.data.dir)
    files = discover(folder, cfg.data.extensions)
    if not files:
        raise SystemExit(f"No documents found in {folder} (looking for {cfg.data.extensions}).")
    all_chunks = []
    for f in files:
        try:
            chunks = chunk_document(load_file(f), cfg.chunking.max_words, cfg.chunking.overlap_words)
        except Exception as e:
            print(f"  ! skipped {f.name}: {e}")
            continue
        for c in chunks:  # global ordinal-safe ids even if two files share a stem
            c.id = f"{c.doc_id}::{c.ordinal:04d}"
        print(f"  {f.name}: {len(chunks)} chunks")
        all_chunks += chunks
    ix = Index.build(all_chunks, cfg)
    out = ix.save()
    print(f"Indexed {len(all_chunks)} chunks from {len(files)} files -> {out}"
          f" ({'hybrid BM25+dense' if ix.dense_enabled else 'BM25 only'})")
    return ix


class RAGEngine:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.reload()

    def reload(self) -> None:
        self.index = Index.load(self.cfg)
        self.retriever = Retriever(self.index, self.cfg)

    def reindex(self) -> None:
        ingest(self.cfg)
        self.reload()

    def retrieve(self, question: str) -> Retrieval:
        return self.retriever.retrieve(question)

    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        t0 = time.perf_counter()
        history = history or []
        query = condense_question(self.cfg, history, question)
        ret = self.retrieve(query)
        t1 = time.perf_counter()

        if ret.abstain:
            return self._pack("I couldn't find this in the loaded documents. " + ret.reason, True, [], "none",
                              query, t0, t1, "")

        if llm_available():
            ans = generate_answer(self.cfg, question, ret.hits, history)
        else:
            ans = extractive_answer(ret.hits)
        return self._pack(ans.text, ans.abstained, ret.hits, ans.mode, query, t0, t1, ans.warning, ans.cited)

    @staticmethod
    def _pack(text, abstained, hits, mode, query, t0, t1, warning, cited=None) -> dict:
        cited = set(cited or [])
        sources = []
        for i, h in enumerate(hits, 1):
            c = h.chunk
            sources.append({
                "id": f"S{i}", "cited": f"S{i}" in cited, "doc_id": c.doc_id, "title": c.title, "file": c.source,
                "clause": c.clause, "section": c.section_path, "page": c.page,
                "snippet": (h.context_text or c.text)[:900], "score": round(h.score, 4),
                "clause_match": h.clause_match})
        return {"answer": text, "abstained": abstained, "mode": mode, "query_used": query, "warning": warning,
                "sources": sources, "timing_ms": {"retrieval": round((t1 - t0) * 1000, 1),
                                                  "total": round((time.perf_counter() - t0) * 1000, 1)}}
