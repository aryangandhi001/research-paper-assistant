"""Chunks a paper's full text, embeds chunks via the Gemini API (not a local
model -- see llm.py's docstring for why), and retrieves the most relevant
chunks for a question -- so Q&A answers are grounded in the actual paper
text rather than the model's general knowledge (and, unlike model
knowledge, works even for very recent papers)."""

import numpy as np

from src.llm import embed_texts

CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


class PaperIndex:
    """An in-memory retrieval index over one paper's chunked text. A vector
    DB would be overkill here -- one paper's chunks (tens, not millions)
    fits comfortably in memory, and cosine similarity over a small numpy
    array is fast enough to not need approximate search.

    Accepts either raw `text` (chunks + embeds it fresh) or precomputed
    `chunks`/`embeddings` (skips both chunking and the Gemini embedding
    API call entirely) -- the latter is what src/cache.py's disk cache
    reconstructs from, so a previously-processed paper never needs its
    chunks re-embedded."""

    def __init__(self, text: str | None = None, chunks: list[str] | None = None, embeddings: np.ndarray | None = None):
        if chunks is not None and embeddings is not None:
            self.chunks = chunks
            self.embeddings = embeddings
        else:
            self.chunks = chunk_text(text)
            self.embeddings = embed_texts(self.chunks)

    def retrieve(self, question: str, k: int = 4) -> list[str]:
        query_vec = embed_texts([question])[0]
        scores = self.embeddings @ query_vec
        top_idx = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in top_idx]


if __name__ == "__main__":
    from src.arxiv_search import search_arxiv
    from src.pdf_extract import extract_full_text

    papers = search_arxiv("reinforcement learning for robot navigation", max_results=1)
    paper = papers[0]
    print(f"Paper: {paper['title']}")
    text = extract_full_text(paper["pdf_url"])

    index = PaperIndex(text)
    print(f"Chunked into {len(index.chunks)} pieces")

    results = index.retrieve("What method did the authors use?", k=3)
    for r in results:
        print("---")
        print(r[:300])
