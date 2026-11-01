from types import SimpleNamespace

from civilrag import generate
from civilrag.config import load_config
from civilrag.pipeline import RAGEngine, ingest


def _fake_client(reply):
    block = SimpleNamespace(type="text", text=reply)
    msgs = SimpleNamespace(create=lambda **kw: SimpleNamespace(content=[block]))
    return lambda: SimpleNamespace(messages=msgs)


def _engine(tmp_path):
    cfg = load_config()
    cfg.embeddings.enabled = False
    cfg.store_dir = str(tmp_path)
    ingest(cfg)
    return RAGEngine(cfg)


def test_llm_answer_and_citation_parsing(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(generate, "_client", _fake_client("Severe exposure needs 45 mm cover [S1]."))
    out = _engine(tmp_path).ask("nominal cover for severe exposure")
    assert out["mode"] == "llm" and not out["abstained"]
    assert out["sources"][0]["cited"] and not out["sources"][1]["cited"]


def test_not_found_is_abstention(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(generate, "_client", _fake_client("NOT_FOUND: fire resistance cover is not covered."))
    out = _engine(tmp_path).ask("nominal cover for severe exposure")
    assert out["abstained"] and "fire resistance" in out["answer"]


def test_uncited_answer_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(generate, "_client", _fake_client("Cover is 45 mm."))
    assert "unverified" in _engine(tmp_path).ask("nominal cover for severe exposure")["warning"]
