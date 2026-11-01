"""Hybrid retrieval: BM25 + dense -> reciprocal-rank fusion -> clause boost -> (rerank) -> neighbours."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .chunking import Chunk
from .config import Config
from .index import Index

CLAUSE_REF = re.compile(r"(?:clause|cl\.?|section|sec\.?|article|para(?:graph)?)\s*(\d+(?:\.\d+)*)|\b(\d+(?:\.\d+)+)\b", re.I)
TABLE_REF = re.compile(r"\btable\s+(\d+(?:\.\d+)*)", re.I)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float = 0.0
    dense: float | None = None
    clause_match: bool = False
    context_text: str = ""          # chunk text plus same-clause neighbours (what the LLM reads)


@dataclass
class Retrieval:
    hits: list[Hit] = field(default_factory=list)
    abstain: bool = False
    reason: str = ""


def parse_refs(query: str) -> tuple[set[str], set[str]]:
    clauses = {a or b for a, b in CLAUSE_REF.findall(query)}
    tables = set(TABLE_REF.findall(query))
    return clauses, tables


class Retriever:
    def __init__(self, index: Index, cfg: Config):
        self.ix, self.cfg, self.r = index, cfg, cfg.retrieval
        self._reranker = None

    def _clause_matches(self, query: str) -> set[int]:
        clauses, tables = parse_refs(query)
        ids: set[int] = set()
        for ref in clauses:
            for cl, idxs in self.ix.by_clause.items():
                if cl == ref or cl.startswith(ref + "."):
                    ids.update(idxs)
        for t in tables:
            pat = re.compile(rf"\btable\s+{re.escape(t)}\b", re.I)
            ids.update(i for i, c in enumerate(self.ix.chunks) if pat.search(c.text))
        return ids

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        try:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.r.rerank_model)
            scores = self._reranker.predict([(query, h.chunk.index_text) for h in hits])
            for h, s in zip(hits, scores):
                h.score = float(s)
            return sorted(hits, key=lambda h: -h.score)
        except Exception as e:  # reranking is an optimisation, never a hard dependency
            print(f"  ! rerank skipped: {e}")
            return hits

    def retrieve(self, query: str) -> Retrieval:
        r, ix = self.r, self.ix
        if not ix.chunks:
            return Retrieval(abstain=True, reason="The index is empty. Add documents and run ingest.")

        bm = ix.search_bm25(query, r.candidates)
        dn = ix.search_dense(query, r.candidates)
        bm_score, dn_score = dict(bm), dict(dn)

        fused: dict[int, float] = {}
        for rank, (i, _) in enumerate(bm):
            fused[i] = fused.get(i, 0.0) + r.bm25_weight / (r.rrf_k + rank + 1)
        for rank, (i, _) in enumerate(dn):
            fused[i] = fused.get(i, 0.0) + r.dense_weight / (r.rrf_k + rank + 1)

        matched = self._clause_matches(query)
        for i in matched:
            fused[i] = fused.get(i, 0.0) + r.clause_boost

        if not fused:
            return Retrieval(abstain=True, reason="No passage shares any terms with the question.")

        ranked = sorted(fused, key=lambda i: -fused[i])
        hits = [Hit(ix.chunks[i], fused[i], bm_score.get(i, 0.0), dn_score.get(i), i in matched) for i in ranked]
        if r.rerank:
            hits = self._rerank(query, hits[: r.candidates])

        # confidence gate: refuse early rather than let the LLM improvise from weak evidence
        top = hits[0]
        if not matched:
            if ix.dense_enabled:
                if max((h.dense or 0.0) for h in hits[:5]) < r.min_dense_sim:
                    return Retrieval(abstain=True, reason="Nothing in the library is close enough to this question.")
            elif max(ix.coverage(query, ix.chunks.index(h.chunk)) for h in hits[:5]) < r.min_coverage:
                return Retrieval(abstain=True, reason="Nothing in the library is close enough to this question.")

        chosen: list[Hit] = []
        per_doc: dict[str, int] = {}
        for h in hits:
            if per_doc.get(h.chunk.doc_id, 0) >= r.max_per_doc:
                continue
            per_doc[h.chunk.doc_id] = per_doc.get(h.chunk.doc_id, 0) + 1
            chosen.append(h)
            if len(chosen) >= r.top_k:
                break

        self._attach_neighbours(chosen)
        return Retrieval(hits=chosen)

    def _attach_neighbours(self, chosen: list[Hit]) -> None:
        chunks = self.ix.chunks
        picked = {h.chunk.id for h in chosen}
        pos = {c.id: i for i, c in enumerate(chunks)}
        for h in chosen:
            parts = [h.chunk.text]
            i = pos[h.chunk.id]
            for step in range(1, self.r.neighbors + 1):
                for j, side in ((i - step, "before"), (i + step, "after")):
                    if 0 <= j < len(chunks):
                        n = chunks[j]
                        if (n.doc_id == h.chunk.doc_id and n.clause == h.chunk.clause and n.clause
                                and n.id not in picked):
                            parts = [n.text] + parts if side == "before" else parts + [n.text]
            h.context_text = "\n\n".join(parts)
