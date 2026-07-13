"""Gradio demo: search for papers on a topic, get an expert-level critical
analysis of any paper, and ask grounded follow-up questions about it."""

import os

import gradio as gr

from src.arxiv_search import search_arxiv
from src.pdf_extract import extract_full_text
from src.rag import PaperIndex
from src.semantic_scholar import enrich_with_citations, rank_papers
from src.summarize import analyze_paper, answer_question, evaluate_summary

# Cache of loaded papers this session: arxiv_id -> {full_text, index}
_paper_cache: dict[str, dict] = {}
_last_search_results: list[dict] = []


def do_search(topic: str):
    global _last_search_results
    if not topic.strip():
        return "Enter a research topic first.", gr.update(choices=[])

    papers = search_arxiv(topic, max_results=12)
    papers = enrich_with_citations(papers)
    papers = rank_papers(papers)
    _last_search_results = papers

    lines = [f"**Top papers for \"{topic}\":**\n"]
    choices = []
    for p in papers:
        cites = p.get("citation_count")
        cite_str = f"{cites} citations" if cites is not None else "citation count unavailable"
        authors = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
        lines.append(
            f"### [{p['title']}]({p['abs_url']})\n"
            f"_{authors} — {p['published']} — {cite_str}_\n\n"
            f"{p['abstract'][:280]}...\n"
        )
        choices.append(p["title"])

    return "\n".join(lines), gr.update(choices=choices, value=choices[0] if choices else None)


def load_paper(title: str, progress=gr.Progress()):
    paper = next((p for p in _last_search_results if p["title"] == title), None)
    if paper is None:
        return "Pick a paper from the search results first.", {}

    if paper["arxiv_id"] not in _paper_cache:
        progress(0.2, desc="Downloading and extracting PDF...")
        full_text = extract_full_text(paper["pdf_url"])
        progress(0.5, desc="Building retrieval index...")
        index = PaperIndex(full_text)
        _paper_cache[paper["arxiv_id"]] = {"full_text": full_text, "index": index, "paper": paper}

    progress(0.7, desc="Generating expert analysis...")
    entry = _paper_cache[paper["arxiv_id"]]
    analysis = analyze_paper(entry["full_text"])
    scores = evaluate_summary(analysis, paper["abstract"])

    progress(1.0)
    footer = f"\n\n---\n_ROUGE vs. authors' own abstract (sanity check, not a quality guarantee): {scores}_"
    return analysis + footer, []


def ask_followup(title: str, question: str, history: list):
    paper = next((p for p in _last_search_results if p["title"] == title), None)
    if paper is None or paper["arxiv_id"] not in _paper_cache:
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Load the paper's analysis first (above) before asking questions."},
        ]
        return history, ""
    if not question.strip():
        return history, ""

    index = _paper_cache[paper["arxiv_id"]]["index"]
    answer, chunks = answer_question(index, question)
    sources = "\n\n".join(f"> {c[:200]}..." for c in chunks[:2])
    full_answer = f"{answer}\n\n<details><summary>Grounding excerpts</summary>\n\n{sources}\n\n</details>"

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": full_answer},
    ]
    return history, ""


with gr.Blocks(title="Research Paper Assistant") as demo:
    gr.Markdown(
        "# Research Paper Discovery & Analysis Assistant\n"
        "Search a topic, get real ranked papers (arXiv + Semantic Scholar citation counts), "
        "then get an expert-level critical analysis of any paper and ask grounded follow-up "
        "questions -- answers are retrieved from the actual paper text, not general knowledge."
    )

    topic_input = gr.Textbox(label="Research topic", placeholder="e.g. reinforcement learning for robot navigation")
    search_btn = gr.Button("Search", variant="primary")
    search_results = gr.Markdown()

    gr.Markdown("---")
    paper_select = gr.Dropdown(label="Select a paper to analyze", choices=[])
    analyze_btn = gr.Button("Analyze paper", variant="primary")
    analysis_output = gr.Markdown()

    gr.Markdown("### Ask follow-up questions about this paper")
    chatbot = gr.Chatbot()
    question_input = gr.Textbox(label="Question", placeholder="e.g. What are the main limitations?")
    ask_btn = gr.Button("Ask")

    search_btn.click(do_search, inputs=topic_input, outputs=[search_results, paper_select])
    analyze_btn.click(load_paper, inputs=paper_select, outputs=[analysis_output, chatbot])
    ask_btn.click(ask_followup, inputs=[paper_select, question_input, chatbot], outputs=[chatbot, question_input])
    question_input.submit(ask_followup, inputs=[paper_select, question_input, chatbot], outputs=[chatbot, question_input])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
