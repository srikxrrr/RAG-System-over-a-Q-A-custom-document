import pytest

from civilrag.chunking import chunk_document
from civilrag.config import load_config
from civilrag.generate import parse_citations
from civilrag.index import tokenize
from civilrag.loaders import Page
from civilrag.pipeline import RAGEngine, ingest
from civilrag.retrieve import parse_refs

DOC = """# Demo Code
## 4 Durability
### 4.2 Nominal cover
Cover shall be 40 mm.

Table 3 Cover by class
| Class | Cover |
|---|---|
| Mild | 20 |
| Severe | 45 |

### 4.3 Water cement ratio
Maximum ratio is 0.5.
"""


def test_chunker_keeps_clauses_and_tables_together():
    chunks = chunk_document([Page("demo", "Demo Code", "demo.md", None, DOC)])
    assert [c.clause for c in chunks] == ["4.2", "4.3"]
    assert "| Severe | 45 |" in chunks[0].text and "Table 3" in chunks[0].text
    assert chunks[0].section_path == "Durability > Nominal cover"


def test_long_table_split_repeats_header():
    rows = "\n".join(f"| r{i} | {i} |" for i in range(80))
    doc = f"## 1 Data\nTable 1 Big\n| a | b |\n|---|---|\n{rows}\n"
    chunks = chunk_document([Page("d", "D", "d.md", None, doc)], max_words=60)
    assert len(chunks) > 1
    assert all("| a | b |" in c.text for c in chunks)


def test_tokenizer_keeps_clause_numbers_and_units():
    t = tokenize("Clause 5.2.1: cover 40mm, water-cement ratio")
    assert "5.2.1" in t and "40" in t and "mm" in t and "water" in t and "cement" in t


def test_parse_refs():
    clauses, tables = parse_refs("What does clause 4.2 and Table 3 say? Also 5.1.2")
    assert {"4.2", "5.1.2"} <= clauses and tables == {"3"}


def test_parse_citations_ignores_invalid_tags():
    assert parse_citations("A [S1][S2, S3]. B [S9].", 3) == ["S1", "S2", "S3"]


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    cfg = load_config()
    cfg.embeddings.enabled = False
    cfg.store_dir = str(tmp_path_factory.mktemp("store"))
    ingest(cfg)
    return RAGEngine(cfg)


def test_exact_clause_lookup_ranks_first(engine):
    ret = engine.retrieve("What does clause 4.3 say about lap splices?")
    assert ret.hits[0].chunk.clause == "4.3" and "lap" in ret.hits[0].chunk.text.lower()


def test_answer_in_top_k(engine):
    ret = engine.retrieve("maximum water-cement ratio very severe exposure")
    assert any("0.40" in h.chunk.text for h in ret.hits)


def test_out_of_scope_abstains(engine):
    assert engine.retrieve("seismic zone factor for Hyderabad").abstain


def test_offline_ask_returns_cited_sources(engine, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = engine.ask("minimum stirrup diameter")
    assert out["mode"] == "extractive" and out["sources"] and out["sources"][0]["cited"]
