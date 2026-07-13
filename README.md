# Research Paper Discovery & Analysis Assistant

Born from a real annoyance during a research internship at RRC, IIIT Hyderabad:
reading through papers to find what's actually relevant is slow. This tool
takes a research topic, finds and ranks real papers on it (via arXiv), and
lets you get a full-paper summary and ask follow-up questions about any paper
— grounded in the actual paper text, not just the abstract.

## How it works

1. **Search** (`src/arxiv_search.py`) — query arXiv's public API for papers
   matching a topic, enriched with real citation counts from Semantic
   Scholar (`src/semantic_scholar.py`) so "top papers" reflects actual
   influence, not just keyword relevance.
2. **Full-text extraction** (`src/pdf_extract.py`) — downloads the paper PDF
   and extracts the full text, not just the abstract. Summarizing only the
   abstract wouldn't save any real reading time — the point is to get the
   substance without reading all 10+ pages.
3. **Retrieval-augmented Q&A** (`src/rag.py`) — chunks the full paper text,
   embeds it locally (sentence-transformers, no API cost), and retrieves the
   most relevant chunks for any question, so answers are grounded in the
   actual paper rather than the model's general knowledge.
4. **Summarization + Q&A generation** (`src/llm.py`, `src/summarize.py`) —
   Google Gemini (free tier).

## Evaluation, not just vibes

Every generated summary is scored with ROUGE against the paper's own
abstract (a reasonable proxy ground truth — if the summary doesn't overlap
with what the authors themselves said the paper is about, that's a real
signal something's off). Retrieved chunks for each Q&A answer are shown
alongside the answer, so grounding can be checked directly rather than
trusted blindly.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Get a free Gemini API key at https://aistudio.google.com/apikey, then:

```bash
set GEMINI_API_KEY=your_key_here      # Windows cmd
$env:GEMINI_API_KEY="your_key_here"   # PowerShell
```

## Usage

```bash
python app.py
```

## Project structure

```
src/
  arxiv_search.py       # arXiv API search + ranking
  semantic_scholar.py   # citation-count enrichment
  pdf_extract.py         # PDF download + full-text extraction
  rag.py                 # chunking, local embeddings, retrieval
  llm.py                 # Gemini API wrapper
  summarize.py           # summary generation + ROUGE evaluation
app.py                   # Gradio demo
```
