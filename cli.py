from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    from .config import load_config
    from .generate import llm_available

    p = argparse.ArgumentParser(prog="civilrag", description="RAG Q&A over civil engineering documents")
    p.add_argument("--config", help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="parse, chunk and index everything in data/raw")
    a = sub.add_parser("ask", help="ask one question")
    a.add_argument("question")
    a.add_argument("--json", action="store_true")
    sub.add_parser("chat", help="interactive multi-turn chat")
    s = sub.add_parser("serve", help="start the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    e = sub.add_parser("eval", help="run the evaluation set")
    e.add_argument("--judge", action="store_true", help="also grade faithfulness with Claude (needs API key)")
    e.add_argument("--qa", default="evals/qa_set.jsonl")
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "ingest":
        from .pipeline import ingest
        ingest(cfg)
    elif args.cmd in ("ask", "chat"):
        from .pipeline import RAGEngine
        eng = RAGEngine(cfg)
        if not llm_available():
            print("(offline mode: set ANTHROPIC_API_KEY for written, cited answers)\n", file=sys.stderr)
        if args.cmd == "ask":
            out = eng.ask(args.question)
            print(json.dumps(out, indent=2, ensure_ascii=False) if args.json else _render(out))
        else:
            hist: list[dict] = []
            print("Ask about your documents. Ctrl-D to exit.")
            while True:
                try:
                    q = input("\n> ").strip()
                except EOFError:
                    break
                if not q:
                    continue
                out = eng.ask(q, hist)
                print(_render(out))
                hist += [{"role": "user", "content": q}, {"role": "assistant", "content": out["answer"]}]
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("civilrag.api:app", host=args.host, port=args.port)
    elif args.cmd == "eval":
        from .evaluate import run_eval
        run_eval(cfg, args.qa, judge=args.judge)


def _render(out: dict) -> str:
    lines = [out["answer"], ""]
    for s in out["sources"]:
        mark = "●" if s["cited"] else "○"
        where = f"cl. {s['clause']}" if s["clause"] else s["section"]
        pg = f", p. {s['page']}" if s["page"] else ""
        lines.append(f"  {mark} [{s['id']}] {s['title']} — {where}{pg}")
    if out["warning"]:
        lines.append(f"\n⚠ {out['warning']}")
    return "\n".join(lines)
