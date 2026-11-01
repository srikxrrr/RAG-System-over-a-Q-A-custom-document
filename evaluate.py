"""Evaluation: retrieval quality, answer-context coverage, abstention, optional LLM faithfulness judge."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import Config
from .generate import _client, format_sources, llm_available
from .pipeline import RAGEngine


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _clause_ok(chunk_clause: str, gold: str | None) -> bool:
    return bool(gold) and (chunk_clause == gold or chunk_clause.startswith(gold + "."))


def judge_faithfulness(cfg: Config, question: str, answer: str, hits) -> float:
    """Claude-as-judge: fraction of the answer's claims that the sources support (0..1)."""
    prompt = (f"{format_sources(hits)}\n\nQuestion: {question}\n\nAnswer to grade:\n{answer}\n\n"
              "List each factual claim in the answer, mark it supported or unsupported by the sources, then end "
              'with a final line of JSON: {"supported": <int>, "total": <int>}')
    r = _client().messages.create(model=cfg.generation.fast_model, max_tokens=700, temperature=0,
                                  messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in r.content if b.type == "text")
    m = re.findall(r'\{"supported":\s*(\d+),\s*"total":\s*(\d+)\}', text)
    if not m:
        return float("nan")
    s, t = map(int, m[-1])
    return s / t if t else 1.0


def run_eval(cfg: Config, qa_path: str = "evals/qa_set.jsonl", judge: bool = False) -> dict:
    eng = RAGEngine(cfg)
    items = [json.loads(line) for line in open(cfg.path(qa_path), encoding="utf-8") if line.strip()]
    k = cfg.retrieval.top_k
    use_llm = llm_available()
    rows = []
    for it in items:
        ret = eng.retrieve(it["question"])
        ctx = _norm(" ".join(h.context_text or h.chunk.text for h in ret.hits))
        row = {"id": it["id"], "question": it["question"], "answerable": it["answerable"], "abstained": ret.abstain}
        if it["answerable"]:
            docs = [h.chunk.doc_id for h in ret.hits]
            row["doc_hit"] = float(it["gold_doc"] in docs)
            ranks = [i for i, h in enumerate(ret.hits, 1)
                     if h.chunk.doc_id == it["gold_doc"] and _clause_ok(h.chunk.clause, it["gold_clause"])]
            row["clause_hit"] = float(bool(ranks))
            row["mrr"] = 1 / ranks[0] if ranks else 0.0
            row["context_recall"] = float(all(_norm(t) in ctx for t in it["must_include"]))
        if use_llm:
            out = eng.ask(it["question"])
            row["abstained"] = out["abstained"]
            if it["answerable"]:
                row["answer_correct"] = float(all(_norm(t) in _norm(out["answer"]) for t in it["must_include"]))
                row["cited"] = float(any(s["cited"] for s in out["sources"]))
                if judge and not out["abstained"]:
                    row["faithfulness"] = judge_faithfulness(cfg, it["question"], out["answer"], ret.hits)
        rows.append(row)

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    avg = lambda key, rs: round(sum(r[key] for r in rs if key in r) / max(1, sum(1 for r in rs if key in r)), 3)
    summary = {
        "questions": len(rows), "top_k": k, "mode": "llm" if use_llm else "retrieval-only",
        "hybrid_dense": eng.index.dense_enabled,
        f"doc_hit@{k}": avg("doc_hit", ans), f"clause_hit@{k}": avg("clause_hit", ans), "MRR": avg("mrr", ans),
        "context_recall": avg("context_recall", ans),
        "answerable_wrongly_abstained": sum(r["abstained"] for r in ans),
        "unanswerable_correctly_abstained": f"{sum(r['abstained'] for r in unans)}/{len(unans)}",
    }
    if use_llm:
        summary["answer_correct"] = avg("answer_correct", ans)
        summary["cited_rate"] = avg("cited", ans)
        if judge:
            summary["faithfulness"] = avg("faithfulness", ans)

    lines = ["# Evaluation report", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k_} | {v} |" for k_, v in summary.items()]
    lines += ["", "## Misses", ""]
    for r in rows:
        bad = (r["answerable"] and (r.get("clause_hit") == 0.0 or r.get("context_recall") == 0.0 or r["abstained"])) \
              or (not r["answerable"] and not r["abstained"])
        if bad:
            lines.append(f"- {r['id']}: {r['question']}  ->  {json.dumps({k_: v for k_, v in r.items() if k_ not in ('id', 'question')})}")
    report = "\n".join(lines)
    out_dir = cfg.path("evals")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(report)
    return summary
