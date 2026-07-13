"""Synthesizes themes and trends *across* multiple papers on a topic --
the actual value of a literature review, not just summarizing one paper at
a time. Uses abstracts (not full text) across many papers, the same way a
researcher skims a field before going deep on any single paper."""

from src.llm import generate

SYNTHESIS_PROMPT = """You are a senior researcher writing the literature-review \
section of a paper on "{topic}", briefing a new student on the field -- not \
listing summaries one after another. Base everything strictly on the \
abstracts provided; if something isn't supported by them, don't claim it.

Structure your synthesis as:

**The core problem**: What underlying problem is this body of work actually \
trying to solve, stated precisely?

**Major approaches**: The distinct families of approaches represented here \
-- group papers by approach, not list them individually. Name what \
meaningfully differs between them.

**How the field has evolved**: Using the publication years, what's the \
actual trajectory -- what did earlier work establish, what did later work \
change or improve on, and why (as far as the abstracts indicate)?

**Key tradeoffs and open disagreements**: Where do these approaches \
genuinely disagree or trade off against each other (e.g. speed vs. \
optimality, centralized vs. decentralized, assumptions that don't always hold)?

**Gaps**: What does this set of papers collectively NOT address, that would \
be a natural next research direction?

--- PAPERS ---
{papers_block}
"""


def synthesize_literature(papers: list[dict], topic: str, max_papers: int = 10) -> str:
    papers = papers[:max_papers]
    papers_block = "\n\n".join(
        f"[{p['published'][:4]}] {p['title']}\n{p['abstract']}"
        for p in papers
    )
    return generate(SYNTHESIS_PROMPT.format(topic=topic, papers_block=papers_block))


if __name__ == "__main__":
    from src.arxiv_search import search_arxiv

    topic = "decentralized multi-agent trajectory planning"
    papers = search_arxiv(topic, max_results=8)
    synthesis = synthesize_literature(papers, topic)
    print(synthesis)
