"""Searches arXiv's public API for papers matching a research topic."""

import urllib.parse

import feedparser

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def search_arxiv(topic: str, max_results: int = 15) -> list[dict]:
    """Returns a list of papers matching `topic`, sorted by relevance:
    [{title, authors, abstract, arxiv_id, pdf_url, abs_url, published,
    categories}, ...]."""
    query = urllib.parse.urlencode({
        "search_query": f"all:{topic}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    feed = feedparser.parse(f"{ARXIV_API_URL}?{query}")

    papers = []
    for entry in feed.entries:
        arxiv_id = entry.id.split("/abs/")[-1]
        pdf_url = next((link.href for link in entry.links if link.type == "application/pdf"), None)

        papers.append({
            "title": " ".join(entry.title.split()),
            "authors": [a.name for a in entry.authors],
            "abstract": " ".join(entry.summary.split()),
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
            "abs_url": entry.id,
            "published": entry.published[:10],
            "categories": [t.term for t in entry.tags] if hasattr(entry, "tags") else [],
        })
    return papers


if __name__ == "__main__":
    papers = search_arxiv("reinforcement learning for robot navigation", max_results=5)
    for p in papers:
        print(f"- {p['title']} ({p['published']}) [{p['arxiv_id']}]")
        print(f"  {p['abstract'][:150]}...")
