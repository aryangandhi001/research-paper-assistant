"""Enriches arXiv papers with real citation counts from Semantic Scholar,
so paper ranking reflects actual influence, not just keyword relevance."""

import requests

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"


def enrich_with_citations(papers: list[dict]) -> list[dict]:
    """Adds a `citation_count` field to each paper (None if not found).
    Uses Semantic Scholar's batch endpoint (one request for all papers)."""
    if not papers:
        return papers

    ids = [f"ARXIV:{p['arxiv_id'].split('v')[0]}" for p in papers]  # strip version suffix
    try:
        resp = requests.post(
            S2_BATCH_URL,
            params={"fields": "citationCount"},
            json={"ids": ids},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json()
    except requests.RequestException:
        results = [None] * len(papers)

    for paper, result in zip(papers, results):
        paper["citation_count"] = result["citationCount"] if result else None

    return papers


def rank_papers(papers: list[dict]) -> list[dict]:
    """Composite ranking: arXiv's relevance order is preserved as the primary
    signal (it already reflects the search query well), with citation count
    as a tiebreaker/influence boost -- a highly-cited paper ranks above a
    similarly-relevant one with few citations."""
    for i, paper in enumerate(papers):
        relevance_rank = len(papers) - i  # earlier = more relevant = higher score
        citation_boost = (paper.get("citation_count") or 0) ** 0.5  # sqrt to avoid citations dominating entirely
        paper["_score"] = relevance_rank + citation_boost
    return sorted(papers, key=lambda p: -p["_score"])


if __name__ == "__main__":
    from src.arxiv_search import search_arxiv

    papers = search_arxiv("reinforcement learning for robot navigation", max_results=8)
    papers = enrich_with_citations(papers)
    papers = rank_papers(papers)
    for p in papers:
        print(f"- [{p.get('citation_count')} cites] {p['title']}")
