"""Downloads a paper's PDF and extracts full text -- summarizing only the
abstract wouldn't save any real reading time over just reading the abstract
itself."""

import io
import re

import pdfplumber
import requests

# Default extraction loses word spacing on many academic-paper PDFs (e.g.
# "modelfreereinforcementlearning") because of how justified/kerned text is
# laid out -- a tighter x_tolerance fixes word-boundary detection.
_X_TOLERANCE = 1.5

_REFERENCES_HEADING = re.compile(r"\n\s*(references|bibliography)\s*\n", re.IGNORECASE)


def extract_full_text(pdf_url: str, max_pages: int = 25) -> str:
    """Downloads the PDF and returns its extracted text (capped at
    `max_pages` -- most papers are well under this; caps runaway extraction
    time on unusually long documents), with the references/bibliography
    section stripped -- it's rarely useful for analysis or Q&A and its
    method-name-heavy citation titles otherwise pollute retrieval, pulling
    in citation text when a question is really about the paper's own method."""
    resp = requests.get(pdf_url, timeout=30)
    resp.raise_for_status()

    text_parts = []
    with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
        for page in pdf.pages[:max_pages]:
            page_text = page.extract_text(x_tolerance=_X_TOLERANCE)
            if page_text:
                text_parts.append(page_text)

    full_text = "\n\n".join(text_parts)

    match = _REFERENCES_HEADING.search(full_text)
    if match:
        full_text = full_text[:match.start()]

    return full_text


if __name__ == "__main__":
    from src.arxiv_search import search_arxiv

    papers = search_arxiv("reinforcement learning for robot navigation", max_results=1)
    paper = papers[0]
    print(f"Extracting: {paper['title']}")
    text = extract_full_text(paper["pdf_url"])
    print(f"Extracted {len(text):,} characters")
    print(text[:500])
