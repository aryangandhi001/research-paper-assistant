"""Generates expert-level paper analysis (not a shallow "this paper is
about X" summary) and grounded Q&A, plus a ROUGE-based sanity check against
the paper's own abstract."""

from rouge_score import rouge_scorer

from src.llm import generate
from src.rag import PaperIndex

ANALYSIS_PROMPT = """You are a senior researcher reviewing this paper for a lab \
meeting, the way an experienced advisor would brief a student -- not writing \
a generic summary. Base everything strictly on the text provided below; if \
something isn't stated in the text, say so rather than guessing.

Structure your analysis as:

**Core contribution**: What is genuinely new here, in one or two sentences? \
Be specific -- not "the authors propose a new method" but what the method \
actually does differently from prior approaches.

**Method**: How does it actually work, concisely but technically -- enough \
that someone could explain it to a colleague.

**Key results**: The main findings and why they matter, not just numbers.

**Critical assessment**: Real strengths and real weaknesses/limitations -- \
assumptions the method depends on, scenarios where it likely wouldn't work, \
gaps in the evaluation. Be genuinely critical, not diplomatically vague.

**Where this sits in the field**: How it relates to or advances prior work \
mentioned in the text.

**Open questions**: What follow-up research this naturally suggests.

--- PAPER TEXT ---
{text}
"""

QA_PROMPT = """You are a senior researcher answering a question about this \
paper, grounded strictly in the excerpts below -- if the excerpts don't \
contain the answer, say so explicitly rather than guessing or relying on \
general knowledge. Answer with the same precision and critical eye a real \
expert would, not a generic restatement.

--- RELEVANT EXCERPTS ---
{context}

--- QUESTION ---
{question}
"""


def analyze_paper(full_text: str, max_chars: int = 30_000) -> str:
    """Full critical analysis of a paper. Truncates very long papers to stay
    within a reasonable prompt size -- most papers' substantive content
    (intro through conclusion) fits well within this budget."""
    return generate(ANALYSIS_PROMPT.format(text=full_text[:max_chars]))


def evaluate_summary(summary: str, abstract: str) -> dict:
    """ROUGE score of the generated analysis against the paper's own
    abstract -- a reasonable proxy ground truth: if the analysis doesn't
    substantially overlap with what the authors themselves said the paper
    is about, that's a real signal something's off, not just a vibe check."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(abstract, summary)
    return {metric: round(score.fmeasure, 3) for metric, score in scores.items()}


def answer_question(index: PaperIndex, question: str, k: int = 4) -> tuple[str, list[str]]:
    """Returns (answer, retrieved_chunks) -- the chunks are surfaced to the
    caller so grounding can be checked directly instead of trusted blindly."""
    chunks = index.retrieve(question, k=k)
    context = "\n\n---\n\n".join(chunks)
    answer = generate(QA_PROMPT.format(context=context, question=question))
    return answer, chunks


if __name__ == "__main__":
    from src.arxiv_search import search_arxiv
    from src.pdf_extract import extract_full_text

    papers = search_arxiv("reinforcement learning for robot navigation", max_results=1)
    paper = papers[0]
    print(f"Paper: {paper['title']}\n")

    text = extract_full_text(paper["pdf_url"])
    analysis = analyze_paper(text)
    print(analysis)

    scores = evaluate_summary(analysis, paper["abstract"])
    print(f"\nROUGE vs. abstract: {scores}")

    index = PaperIndex(text)
    answer, chunks = answer_question(index, "What are the main limitations of this approach?")
    print(f"\nQ: What are the main limitations of this approach?\nA: {answer}")
