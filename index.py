"""Search index: BM25 (always) + dense embeddings (optional), persisted to `store/`."""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .config import Config

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")
_STOP = set("""a an the of to in on for and or is are was were be been by with as at from that this these those it its
what which who how when where does do did can shall should may must will than then into per""".split())


def tokenize(text: str) -> list[str]:
    """Lower-case tokens that keep clause numbers (5.2.1), grades (m25) and split '40mm' -> '40', 'mm'."""
    text = re.sub(r"(?<=\d)(?=[a-z])", " ", text.lower())
    out: list[str] = []
    for tok in _TOKEN.findall(text):
        if tok in _STOP:
            continue
        out.append(tok)
        if "-" in tok or "/" in tok:  # water-cement -> water, cement as well
            out += [p for p in re.split(r"[-/]", tok) if p and p not in _STOP]
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.dl = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.dl.mean()) if self.n else 0.0
        self.inv: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                self.inv[t].append((i, f))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.inv.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.n)
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.inv[t]:
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg))
        return s


class Embedder:
    """Thin wrapper over sentence-transformers; loaded lazily so the base install stays light."""

    def __init__(self, model: str, query_prefix: str = ""):
        self.model_name, self.prefix, self._m = model, query_prefix, None

    @staticmethod
    def available() -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except Exception:
            return False

    def _model(self):
        if self._m is None:
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(self.model_name)
        return self._m

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model().encode(texts, normalize_embeddings=True, batch_size=32,
                                               show_progress_bar=len(texts) > 64), dtype=np.float32)

    def encode_query(self, q: str) -> np.ndarray:
        return np.asarray(self._model().encode([self.prefix + q], normalize_embeddings=True), dtype=np.float32)[0]


class Index:
    def __init__(self, chunks: list[Chunk], cfg: Config, emb: np.ndarray | None = None):
        self.cfg, self.chunks, self.emb = cfg, chunks, emb
        self.bm25 = BM25([tokenize(c.index_text) for c in chunks])
        self.by_clause: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            if c.clause:
                self.by_clause[c.clause].append(i)
        self._tokset: list[set[str]] | None = None
        self.embedder = (Embedder(cfg.embeddings.model, cfg.embeddings.query_prefix)
                         if emb is not None else None)

    # ---- build / persist -------------------------------------------------
    @classmethod
    def build(cls, chunks: list[Chunk], cfg: Config) -> "Index":
        emb = None
        if cfg.embeddings.enabled and chunks:
            if Embedder.available():
                print(f"  embedding {len(chunks)} chunks with {cfg.embeddings.model} ...")
                emb = Embedder(cfg.embeddings.model).encode_docs([c.index_text for c in chunks])
            else:
                print("  ! sentence-transformers not installed -> BM25-only (pip install sentence-transformers)")
        return cls(chunks, cfg, emb)

    def save(self) -> Path:
        d = self.cfg.path(self.cfg.store_dir)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        if self.emb is not None:
            np.save(d / "embeddings.npy", self.emb)
        elif (d / "embeddings.npy").exists():
            (d / "embeddings.npy").unlink()
        meta = {"n_chunks": len(self.chunks), "embed_model": self.cfg.embeddings.model if self.emb is not None else None,
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "docs": sorted({c.doc_id for c in self.chunks})}
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return d

    @classmethod
    def load(cls, cfg: Config) -> "Index":
        d = cfg.path(cfg.store_dir)
        if not (d / "chunks.jsonl").exists():
            raise FileNotFoundError("No index found. Run `python -m civilrag ingest` first.")
        chunks = [Chunk(**json.loads(line)) for line in open(d / "chunks.jsonl", encoding="utf-8")]
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        emb = None
        if (d / "embeddings.npy").exists():
            if meta.get("embed_model") == cfg.embeddings.model and Embedder.available():
                emb = np.load(d / "embeddings.npy")
            else:
                print("  ! stored embeddings unusable (model changed or sentence-transformers missing) -> BM25-only")
        return cls(chunks, cfg, emb)

    # ---- search ----------------------------------------------------------
    def search_bm25(self, query: str, n: int) -> list[tuple[int, float]]:
        s = self.bm25.scores(tokenize(query))
        order = np.argsort(-s)[:n]
        return [(int(i), float(s[i])) for i in order if s[i] > 0]

    def search_dense(self, query: str, n: int) -> list[tuple[int, float]]:
        if self.emb is None or self.embedder is None:
            return []
        sims = self.emb @ self.embedder.encode_query(query)
        order = np.argsort(-sims)[:n]
        return [(int(i), float(sims[i])) for i in order]

    def coverage(self, query: str, chunk_idx: int) -> float:
        """Share of the query's IDF mass found in one chunk (query words absent from the whole corpus
        count at maximum weight). Independent of query length, unlike a raw BM25 score."""
        if self._tokset is None:
            self._tokset = [set(tokenize(c.index_text)) for c in self.chunks]
        max_idf = math.log(1 + (self.bm25.n + 0.5) / 0.5)
        toks = set(tokenize(query))
        if not toks:
            return 0.0
        total = sum(self.bm25.idf.get(t, max_idf) for t in toks)
        hit = sum(self.bm25.idf[t] for t in toks if t in self._tokset[chunk_idx] and t in self.bm25.idf)
        return hit / total if total else 0.0

    @property
    def dense_enabled(self) -> bool:
        return self.emb is not None
