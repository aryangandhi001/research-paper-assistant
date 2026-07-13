"""Searches arXiv's public API for papers matching a research topic."""

import urllib.parse

import feedparser

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def _fetch(search_query: str, max_results: int) -> list[dict]:
    query = urllib.parse.urlencode({
        "search_query": search_query,
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


def search_arxiv(topic: str, max_results: int = 15) -> list[dict]:
    """Returns a list of papers matching `topic`, sorted by relevance:
    [{title, authors, abstract, arxiv_id, pdf_url, abs_url, published,
    categories}, ...].

    Runs a title-focused query first, then a general all-fields query, and
    merges (title matches ranked first, deduped). A plain all-fields search
    alone performs badly for short/ambiguous paper-name acronyms (e.g.
    "SPOT", "PUMA", "SUPER" are also common English words / used by many
    unrelated papers) -- if a query word literally appears in a paper's own
    title, that's a much stronger relevance signal than it appearing
    somewhere in the abstract of an unrelated, more "keyword-dense" paper,
    which is what a plain relevance-ranked all-fields search tends to surface
    instead for short queries."""
    words = topic.split()
    title_query = " AND ".join(f"ti:{w}" for w in words)
    all_query = f"all:{topic}"

    title_papers = _fetch(title_query, max_results)
    general_papers = _fetch(all_query, max_results)

    seen = set()
    merged = []
    for paper in title_papers + general_papers:
        if paper["arxiv_id"] not in seen:
            seen.add(paper["arxiv_id"])
            merged.append(paper)

    return merged[:max_results]


if __name__ == "__main__":
    for topic in ["SPOT spatio-temporal obstacle free trajectory", "PUMA uncertainty aware trajectory"]:
        papers = search_arxiv(topic, max_results=5)
        print(f"--- {topic} ---")
        for p in papers:
            print(f"- {p['title']} ({p['published']}) [{p['arxiv_id']}]")
        print()
