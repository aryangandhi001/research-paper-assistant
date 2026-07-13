"""Disk-backed cache for a paper's extracted text, chunk embeddings, and
generated analysis, keyed by arxiv_id -- avoids redundant PDF downloads,
Gemini embedding calls, and Gemini generation calls for a paper already
processed, both within one running process and across restarts where the
underlying disk survives.

Honest limitation, stated directly rather than overclaimed: Render's free
tier has no *guaranteed* persistent disk across a full redeploy without a
paid disk add-on. This cache reliably avoids redundant work within one
running instance's lifetime (e.g. re-opening the same paper twice in one
session, or two different users hitting the same warm instance); whether
it also survives an idle spin-down-and-respawn cycle (as opposed to a full
redeploy) depends on Render's own free-tier disk behavior, which isn't
verified for this specific service. Worst case, a cache miss just falls
back to the original from-scratch behavior -- this is a pure optimization,
never a correctness dependency.
"""

import pickle
from pathlib import Path

CACHE_DIR = Path("paper_cache")


def _cache_path(arxiv_id: str) -> Path:
    safe_id = arxiv_id.replace("/", "_")
    return CACHE_DIR / f"{safe_id}.pkl"


def load_cached(arxiv_id: str) -> dict | None:
    """Returns the cached {full_text, chunks, embeddings, analysis} dict for
    this paper, or None on a cache miss or any read/unpickle failure (a
    corrupted or missing cache file degrades to a normal cache miss, not a
    crash)."""
    path = _cache_path(arxiv_id)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_cached(arxiv_id: str, data: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    with open(_cache_path(arxiv_id), "wb") as f:
        pickle.dump(data, f)
