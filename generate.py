"""Answer synthesis with Claude: grounded, cited, and willing to say 'not found'."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .config import Config
from .retrieve import Hit

SYSTEM_PROMPT = """You are a careful reference assistant for civil engineering documents: codes, standards, specifications and course notes.

Answer ONLY from the numbered sources provided in the user message. Do not use outside knowledge.

Rules:
1. Cite every factual statement with its source tag, e.g. [S1] or [S1][S3]. Put the tag at the end of the sentence it supports.
2. Reproduce numbers, units, limits, symbols and formulas exactly as written in the source. Never round, convert units, interpolate or extrapolate unless the source itself does.
3. State the conditions attached to a value (grade, exposure class, member type, edition) together with the value.
4. If sources disagree or come from different documents/editions, say so and cite each.
5. If the sources do not contain the answer, reply with exactly `NOT_FOUND:` followed by one sentence saying what is missing. A partial answer is fine if you say which part is missing.
6. Be concise. Use a small markdown table when giving values for several cases. Refer to clause or table numbers when the source gives them."""

_CITE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]")


@dataclass
class Answer:
    text: str
    abstained: bool
    cited: list[str]          # e.g. ["S1", "S3"]
    mode: str                 # "llm" | "extractive"
    warning: str = ""


def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _client():
    import anthropic
    return anthropic.Anthropic()


def format_sources(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        attrs = f'id="S{i}" document="{c.title}"'
        if c.clause:
            attrs += f' clause="{c.clause}"'
        if c.page:
            attrs += f' page="{c.page}"'
        if c.section_path:
            attrs += f' section="{c.section_path}"'
        blocks.append(f"<source {attrs}>\n{h.context_text or c.text}\n</source>")
    return "\n\n".join(blocks)


def parse_citations(text: str, n_sources: int) -> list[str]:
    seen: list[str] = []
    for group in _CITE.findall(text):
        for tag in re.split(r"[,;]\s*", group):
            tag = tag.strip()
            if int(tag[1:]) <= n_sources and tag not in seen:
                seen.append(tag)
    return seen


def condense_question(cfg: Config, history: list[dict], question: str) -> str:
    """Rewrite a follow-up ('what about severe exposure?') into a standalone search query."""
    if not history or not llm_available():
        return question
    turns = history[-2 * cfg.generation.history_turns:]
    convo = "\n".join(f"{t['role']}: {t['content'][:600]}" for t in turns)
    try:
        r = _client().messages.create(
            model=cfg.generation.fast_model, max_tokens=120, temperature=0,
            system="Rewrite the final user question as one standalone search query, resolving pronouns and "
                   "omitted context from the conversation. Output only the query.",
            messages=[{"role": "user", "content": f"{convo}\nuser: {question}"}])
        out = "".join(b.text for b in r.content if b.type == "text").strip()
        return out or question
    except Exception:
        return question


def generate_answer(cfg: Config, question: str, hits: list[Hit], history: list[dict] | None = None) -> Answer:
    msgs = [{"role": t["role"], "content": t["content"]}
            for t in (history or [])[-2 * cfg.generation.history_turns:]
            if t.get("role") in ("user", "assistant")]
    msgs.append({"role": "user", "content": f"{format_sources(hits)}\n\nQuestion: {question}"})
    try:
        r = _client().messages.create(model=cfg.generation.model, max_tokens=cfg.generation.max_tokens,
                                      temperature=cfg.generation.temperature, system=SYSTEM_PROMPT, messages=msgs)
    except Exception as e:  # network / auth / quota -> degrade gracefully
        ans = extractive_answer(hits)
        ans.warning = f"LLM call failed ({type(e).__name__}); showing retrieved passages instead."
        return ans
    text = "".join(b.text for b in r.content if b.type == "text").strip()
    if text.startswith("NOT_FOUND"):
        return Answer(text.split(":", 1)[-1].strip() or "Not found in the provided documents.", True, [], "llm")
    cited = parse_citations(text, len(hits))
    warn = "" if cited else "The model did not cite any source; treat this answer as unverified."
    return Answer(text, False, cited, "llm", warn)


def extractive_answer(hits: list[Hit]) -> Answer:
    """No-LLM fallback: show the best passages verbatim with their citations."""
    parts = []
    for i, h in enumerate(hits[:3], 1):
        body = h.chunk.text.strip()
        parts.append(f"{body[:600]}{'…' if len(body) > 600 else ''} [S{i}]")
    return Answer("Most relevant passages (offline mode, no Claude API key):\n\n" + "\n\n".join(parts),
                  False, [f"S{i}" for i in range(1, min(3, len(hits)) + 1)], "extractive")
