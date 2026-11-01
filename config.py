from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass
class DataCfg:
    dir: str = "data/raw"
    extensions: list = field(default_factory=lambda: [".pdf", ".md", ".txt", ".docx"])


@dataclass
class ChunkCfg:
    max_words: int = 220
    overlap_words: int = 30


@dataclass
class EmbedCfg:
    enabled: bool = True
    model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "


@dataclass
class RetrievalCfg:
    top_k: int = 5
    candidates: int = 30
    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    clause_boost: float = 0.03
    neighbors: int = 1
    max_per_doc: int = 4
    rerank: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"
    min_coverage: float = 0.25
    min_dense_sim: float = 0.35


@dataclass
class GenCfg:
    model: str = "claude-sonnet-5-5"
    fast_model: str = "claude-haiku-5-5"
    max_tokens: int = 900
    temperature: float = 0.0
    history_turns: int = 4


@dataclass
class Config:
    project: str = "civilcode-rag"
    store_dir: str = "store"
    data: DataCfg = field(default_factory=DataCfg)
    chunking: ChunkCfg = field(default_factory=ChunkCfg)
    embeddings: EmbedCfg = field(default_factory=EmbedCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    generation: GenCfg = field(default_factory=GenCfg)
    root: Path = field(default_factory=lambda: DEFAULT_CONFIG.parent)

    def path(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else self.root / q


_SECTIONS = {
    "data": DataCfg,
    "chunking": ChunkCfg,
    "embeddings": EmbedCfg,
    "retrieval": RetrievalCfg,
    "generation": GenCfg,
}


def load_config(path: str | Path | None = None) -> Config:
    import os

    path = Path(path or os.environ.get("CIVILRAG_CONFIG") or DEFAULT_CONFIG)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = Config(root=path.resolve().parent)
    for key, val in raw.items():
        if key in _SECTIONS:
            setattr(cfg, key, _SECTIONS[key](**(val or {})))  # unknown keys raise -> typos are caught
        elif key in ("project", "store_dir"):
            setattr(cfg, key, val)
        else:
            raise ValueError(f"Unknown config key: {key}")
    return cfg
