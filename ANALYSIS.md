# Research Paper Discovery & Analysis Assistant — Full Technical Report

A complete, exhaustive walkthrough: what this project does, why every
design decision was made, every function in the codebase, every bug hit
and how it was diagnosed and fixed, how it's deployed, and what's honestly
still missing.

---

## 1. Motivation, and why the scope is what it is

This project came out of a genuine, specific frustration: during a
robotics/UAV autonomy research internship (RRC, IIIT Hyderabad), staying
on top of the literature — trajectory planning, multi-agent coordination,
uncertainty-aware planning — meant constantly reading papers, and a lot of
that reading was slow and repetitive: finding the relevant papers in the
first place, then reading each one closely enough to actually judge
whether its contribution mattered or was just incremental.

The original idea for this project was much bigger — an autonomous
GPT-style research agent that could also edit code. That scope was
deliberately cut down after concluding it would mostly amount to vibe-coded
breadth without depth: a large surface area assembled by prompting rather
than something with real engineering decisions and debugging in it. What's
built instead is narrower and does two concrete things well:

1. **Literature synthesis** — search a topic, get a real literature-review-style
   picture of the *field*, not a list of separately-summarized papers.
2. **Single-paper deep analysis** — pick one paper, get a genuinely critical
   expert-level breakdown of it, then ask grounded follow-up questions
   answered strictly from that paper's own text (RAG), not the model's
   general knowledge.

Both of these map directly onto what was actually annoying about reading
papers during the internship: finding out fast whether a paper is worth
reading closely, and then, if it is, being able to interrogate it instead
of re-reading it to find one detail.

---

## 2. Architecture overview

```
src/
  arxiv_search.py      -- finds candidate papers on a topic (arXiv API)
  semantic_scholar.py  -- enriches with real citation counts, re-ranks
  pdf_extract.py        -- downloads + extracts clean full text from a paper's PDF
  rag.py                 -- chunks + embeds + retrieves relevant passages from one paper
  llm.py                 -- thin Gemini API wrapper (generation + embeddings)
  summarize.py           -- single-paper critical analysis + grounded Q&A + ROUGE sanity check
  synthesize.py           -- cross-paper literature synthesis
app.py                   -- Gradio demo, 2 tabs
```

Pipeline for a session:

```
topic --> search_arxiv --> enrich_with_citations --> rank_papers
      --> [Literature synthesis tab]: synthesize_literature(all found papers)
      --> [Single-paper tab]: pick one --> extract_full_text --> PaperIndex
              --> analyze_paper (critical analysis) --> evaluate_summary (ROUGE sanity check)
              --> ask_followup --> PaperIndex.retrieve --> answer_question (grounded Q&A)
```

---

## 3. File-by-file, function-by-function walkthrough

### `src/arxiv_search.py`

```python
def search_arxiv(topic: str, max_results: int = 15) -> list[dict]:
    words = topic.split()
    title_query = " AND ".join(f"ti:{w}" for w in words)
    all_query = f"all:{topic}"
    title_papers = _fetch(title_query, max_results)
    general_papers = _fetch(all_query, max_results)
    # merge, title-matches first, dedup by arxiv_id
```
Queries arXiv's public API twice per search and merges: once restricted to
the **title** field (`ti:word AND ti:word...`), once across all fields
(`all:{topic}`), with title-matches taking priority in the merged,
deduplicated result. `_fetch` does the actual HTTP call via `feedparser`
(arXiv's API returns an Atom feed) and normalizes each entry into a plain
dict: title (whitespace-collapsed), authors, abstract, arxiv_id, pdf_url,
abs_url, published date, categories.

The reason for the dual-query merge is a real, verified bug — covered in
detail in the debugging section — where a plain relevance-ranked
all-fields search performs badly for short/ambiguous paper acronyms.

### `src/semantic_scholar.py`

```python
def enrich_with_citations(papers):
    ids = [f"ARXIV:{p['arxiv_id'].split('v')[0]}" for p in papers]
    resp = requests.post(S2_BATCH_URL, params={"fields": "citationCount"}, json={"ids": ids})
```
Adds real citation counts to each paper via Semantic Scholar's batch
endpoint — a single POST request scores every paper found in the arXiv
search at once, rather than one request per paper. The arXiv version
suffix (`v1`, `v2`, ...) is stripped from the ID before querying, since
Semantic Scholar indexes by the base arXiv ID. Wrapped in a try/except so
a Semantic Scholar outage degrades to `citation_count: None` for every
paper rather than crashing the whole search.

```python
def rank_papers(papers):
    for i, paper in enumerate(papers):
        relevance_rank = len(papers) - i
        citation_boost = (paper.get("citation_count") or 0) ** 0.5
        paper["_score"] = relevance_rank + citation_boost
```
Composite ranking: arXiv's own relevance order is kept as the dominant
signal (papers are already well-ordered by the search itself), with
citation count added as a **tiebreaker/influence boost**, not the primary
sort key — a highly-cited paper should be able to edge out a similarly
relevant one with few citations, but citation count alone shouldn't be
able to override genuine topical relevance (an old, highly-cited but only
tangentially related paper shouldn't bury a directly-on-topic new one).
The square root on citation count is deliberate damping: without it, a
paper with 10,000 citations would completely dominate the score regardless
of relevance, which isn't the intended behavior.

### `src/pdf_extract.py`

```python
_X_TOLERANCE = 1.5

def extract_full_text(pdf_url, max_pages=25):
    ...
    page_text = page.extract_text(x_tolerance=_X_TOLERANCE)
    ...
    match = _REFERENCES_HEADING.search(full_text)
    if match:
        full_text = full_text[:match.start()]
```
Downloads the PDF and extracts text page by page via `pdfplumber`, capped
at 25 pages (generous headroom over a typical paper's length, while
bounding worst-case extraction time on unusually long documents). Two
deliberate, bug-driven details:

- `x_tolerance=1.5` — pdfplumber's default word-spacing heuristic merges
  adjacent words together on many academic PDFs (justified/kerned text
  layouts confuse the default tolerance), covered below.
- Everything from a "References"/"Bibliography" heading onward is
  regex-stripped before returning, because that section is dead weight for
  both the analysis prompt (wastes context budget on citation-list text)
  and, more importantly, for retrieval — covered below.

### `src/llm.py`

```python
MODEL_NAME = "gemini-flash-lite-latest"
EMBEDDING_MODEL_NAME = "gemini-embedding-001"

def embed_texts(texts, model=EMBEDDING_MODEL_NAME) -> np.ndarray:
    result = client.models.embed_content(model=model, contents=texts)
    vectors = np.array([e.values for e in result.embeddings], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-8, None)
```
A deliberately thin wrapper. Two decisions worth calling out explicitly:

1. **Both generation and embeddings go through the Gemini API — no local
   model at all**, not even for embeddings. This is a direct consequence
   of the Render free-tier 512MB RAM limit (covered in the debugging
   section): `sentence-transformers`/`torch` alone is memory-heavy enough
   to get OOM-killed mid-request on that tier, so embeddings were moved
   entirely off-box onto Gemini's own embedding endpoint.
2. **`np.clip(norms, 1e-8, None)`** before dividing — a plain division by
   `norms` would raise a divide-by-zero / produce `nan` for a
   theoretically-possible all-zero embedding vector; clipping the
   denominator's minimum value avoids that without needing a branch.

### `src/rag.py`

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]
```
Simple fixed-size sliding-window chunking (1000 characters, 200-character
overlap between consecutive chunks). The overlap exists so a sentence or
idea that happens to fall right on a chunk boundary isn't cleanly severed
out of both chunks — it's very likely to appear whole in at least one of
the two overlapping chunks around that boundary.

```python
class PaperIndex:
    def __init__(self, text):
        self.chunks = chunk_text(text)
        self.embeddings = embed_texts(self.chunks)

    def retrieve(self, question, k=4):
        query_vec = embed_texts([question])[0]
        scores = self.embeddings @ query_vec
        top_idx = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in top_idx]
```
An in-memory retrieval index scoped to a single paper. Deliberately no
vector database: one paper's chunk count is tens, not millions, so a plain
numpy dot-product search over an in-memory array is both simpler and fast
enough — reaching for a vector DB here would be unjustified complexity for
the actual scale involved. Since `embed_texts` already returns
L2-normalized vectors, `self.embeddings @ query_vec` is exactly cosine
similarity, computed for every chunk against the query in a single
vectorized matrix-vector product.

### `src/summarize.py`

```python
ANALYSIS_PROMPT = """... Be genuinely critical, not diplomatically vague. ..."""
```
The analysis prompt is structured into six explicit sections — Core
contribution, Method, Key results, Critical assessment, Where this sits in
the field, Open questions — and explicitly instructs the model to state
when the source text doesn't support a claim rather than filling gaps with
guesses, and to be "genuinely critical, not diplomatically vague." This
phrasing was chosen deliberately: LLMs left to their own devices tend to
produce diplomatically hedged, uniformly-positive paper summaries; naming
the failure mode directly in the prompt measurably pushes the output
toward actually naming weaknesses and assumptions, which is the entire
point of an analysis tool (a summary that never criticizes anything isn't
useful for deciding whether a paper is worth trusting).

```python
def analyze_paper(full_text, max_chars=30_000):
    return generate(ANALYSIS_PROMPT.format(text=full_text[:max_chars]))
```
Truncates to the first 30,000 characters before sending to the model — a
paper's substantive content (introduction through conclusion/discussion)
comfortably fits this budget for the vast majority of papers, and this
keeps a hard, predictable cap on prompt size regardless of how long any
given PDF's extracted text turns out to be.

```python
def evaluate_summary(summary, abstract):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(abstract, summary)
```
A sanity check, explicitly labeled as such in the UI (not a quality
guarantee): ROUGE-1/2/L overlap between the generated analysis and the
paper's own abstract. The reasoning: the authors' own abstract is a
reasonable proxy ground truth for "what is this paper actually about" — if
the generated analysis has near-zero overlap with it, that's a real signal
something went wrong (wrong paper loaded, garbled PDF extraction, model
hallucinating), even though high overlap by itself doesn't prove the
*critical* parts of the analysis are any good, hence the "sanity check, not
a quality guarantee" framing surfaced directly to the user.

```python
def answer_question(index, question, k=4):
    chunks = index.retrieve(question, k=k)
    context = "\n\n---\n\n".join(chunks)
    answer = generate(QA_PROMPT.format(context=context, question=question))
    return answer, chunks
```
Returns the retrieved chunks alongside the answer, not just the answer
itself — so grounding can be checked directly by a user (the app surfaces
these as a collapsible "Grounding excerpts" section) rather than having to
trust the model's claim that it read the paper correctly.

### `src/synthesize.py`

```python
SYNTHESIS_PROMPT = """... briefing a new student on the field -- not listing
summaries one after another. ..."""

def synthesize_literature(papers, topic, max_papers=10):
    papers = papers[:max_papers]
    papers_block = "\n\n".join(f"[{p['published'][:4]}] {p['title']}\n{p['abstract']}" for p in papers)
    return generate(SYNTHESIS_PROMPT.format(topic=topic, papers_block=papers_block))
```
This is the feature that answers the actual internship-era frustration
most directly: not "summarize each paper" (which is what single-paper
analysis already does, one at a time) but "tell me the shape of this whole
subfield" — core problem, the distinct *families* of approaches (grouped,
not listed paper-by-paper), how the field's actually evolved using
publication years as a real signal, where approaches genuinely trade off
against each other, and what's collectively missing. Deliberately uses
**abstracts only, across up to 10 papers**, not full text — this mirrors
how a researcher actually skims a field (scanning many abstracts) before
committing time to reading any single paper closely, and keeps the prompt
a fixed, small size regardless of how many papers were found.

### `app.py`

Two tabs share one search bar and one cached search-result state
(`_last_search_results`, `_last_topic`, plus a `_paper_cache` dict keyed
by arxiv_id so re-analyzing an already-loaded paper doesn't re-download
and re-embed it):

- **Literature synthesis** — one button, calls `synthesize_literature`
  directly on the cached search results.
- **Single-paper analysis** — a dropdown of found papers, an "Analyze"
  button (downloads/extracts/embeds on first load, cached after), and a
  `gr.Chatbot` for grounded follow-up Q&A.

```python
history = history + [
    {"role": "user", "content": question},
    {"role": "assistant", "content": full_answer},
]
```
Chat history is built as a list of role/content message dicts, the format
Gradio 6.x's `Chatbot` component expects — this specific detail broke
during development and is covered below.

---

## 4. The real debugging journey

### Bug: Gemini free-tier quota was 0 for the "obvious" default model

The first model tried, `gemini-2.0-flash`, returned `RESOURCE_EXHAUSTED`
errors on essentially every request, and inspecting the error detail
showed `limit: 0` — the free tier's per-model quota for that specific
model was zero, not merely low. Diagnosed by explicitly listing available
models against the account (`client.models.list()`) and testing each one's
free-tier quota rather than assuming the "default"/flagship model name
would work. `gemini-flash-lite-latest` had a real, usable free quota and
became the model actually used for all generation calls.

### Bug: pdfplumber merging words together on academic PDFs

Extracted text from several papers came back with words run together —
e.g. `"modelfreereinforcementlearning"` instead of `"model free
reinforcement learning"`. Root cause: pdfplumber's default word-boundary
detection relies on horizontal gaps between characters to decide where one
word ends and the next begins, and many academic paper PDFs use
justified/kerned text layouts where the actual gap between two adjacent
words can be nearly as small as the gap between two letters within a word
— the default tolerance wasn't picking up on real word boundaries.
**Fix:** passing a tighter `x_tolerance=1.5` to `extract_text` made
pdfplumber sensitive enough to the smaller true word-gaps in these layouts
to preserve spacing correctly.

### Bug: references section polluting both analysis and retrieval

Early testing of the Q&A feature surfaced a specific failure: asking a
genuine question about the paper's own method sometimes returned an answer
built from **citation text** — other papers' titles and author lists from
the references section — rather than the actual paper's content. Root
cause: the references/bibliography section of a paper is dense with
method-name-like text (other papers' titles), which made it a plausible
embedding match for method-related questions purely on lexical/semantic
overlap, even though it's never the right thing to retrieve. It was also
wasting real budget out of the 30,000-character analysis truncation for no
benefit. **Fix:** a regex (`_REFERENCES_HEADING`) detects the
"References"/"Bibliography" heading and truncates everything from that
point onward, before the text is either chunked for retrieval or sent to
the analysis prompt.

### Bug: arXiv search failing for short/ambiguous paper acronym names

This was the most personally significant bug, raised directly against
papers relevant to the internship work by name: *"there is paper related
my work like MIGHTY, mader, RMADER, PUMA, uncertainty aware, SPOT,
SUPER...etc... so there are no data related it."* Testing confirmed it —
searching for short acronym-style paper names (SPOT, PUMA, SUPER) using a
plain arXiv all-fields relevance search returned mostly irrelevant results,
because these short strings are also common English words or substrings
that appear incidentally in many unrelated papers' abstracts, and a
keyword-dense unrelated paper could out-rank the actual paper the acronym
names. **Fix:** `search_arxiv` now runs a **title-focused** query
(`ti:word AND ti:word...`) first and merges it ahead of the general
all-fields query — if a query word literally appears in a paper's own
title, that's a far stronger relevance signal than incidental appearance
in an unrelated abstract. Verified directly: searching for
`"SPOT spatio-temporal obstacle free trajectory"` now correctly surfaces
the actual internship paper — *"SPOT: Spatio-Temporal Obstacle-free
Trajectory Planning for UAVs..."* by Astik Srivastava, Thomas J
Chackenkulam, Bitla Bhanu Teja, Antony Thomas, and Madhava Krishna — as a
top result.

### Bug: Gradio 6.x Chatbot message-format mismatch

An early implementation of the chat history passed the old Gradio
tuple-based format (`[(user_msg, bot_msg), ...]`) and/or attempted to pass
`type="messages"` as a kwarg to `gr.Chatbot()`. In the installed Gradio 6.x
version, that kwarg doesn't exist at all — passing it raised a `TypeError`
outright rather than silently doing the wrong thing. **Fix:** switched the
chat history representation itself to the message-dict format Gradio 6.x's
`Chatbot` expects natively — `{"role": "user"/"assistant", "content": ...}`
— with no `type=` kwarg needed at all, since that's simply the component's
default expected shape in this version.

### Bug: OOM crash on Render's free tier (512MB RAM)

The app worked locally but crashed with an out-of-memory kill on Render's
free tier specifically, traced to loading `sentence-transformers` (and its
`torch` dependency) for local embedding generation — `torch` alone,
even before loading any actual model weights, claims enough memory
headroom that it was enough to blow past the 512MB limit under real
request load. **Fix:** removed `torch`, `sentence-transformers`, and
`scikit-learn` from `requirements.txt` entirely and switched embedding
generation to Gemini's own `embed_content` API (see `llm.py` above) — no
local model of any kind runs in the deployed process anymore, which is
also why `llm.py`'s docstring states the memory reasoning directly rather
than leaving it as an unexplained design choice.

### Bug: `GEMINI_API_KEY` not actually applying on Render

After deployment, every generation call failed with "Set the
GEMINI_API_KEY environment variable" even though the variable had been
included in the service's creation-time `envVars` payload via Render's
REST API. Root cause: Render's service-creation endpoint doesn't reliably
apply `envVars` passed at creation time in this workflow. **Fix:** set the
key via a separate, dedicated `PUT /v1/services/{id}/env-vars` call *after*
the service already existed, then manually triggered a new deploy so the
running instance actually picked up the newly-set variable (Render doesn't
auto-redeploy on an out-of-band env-var change via this path). This same
root issue was independently hit and fixed the same way on the F1 predictor
project.

---

## 5. Deployment

Deployed on Render's free tier as a Gradio web service. `requirements.txt`
deliberately excludes `torch`/`sentence-transformers`/`scikit-learn` (see
OOM bug above) — the only heavy dependencies are `pdfplumber` (PDF
extraction), `feedparser` (arXiv Atom feed parsing), `requests`,
`rouge-score`, `google-genai`, `numpy`, and `gradio` itself. `GEMINI_API_KEY`
is set as a Render environment variable via the dedicated env-vars API
endpoint (see above), never committed to the repo.

---

## 6. Honest limitations and what's actually missing

- **`max_pages=25` and `max_chars=30_000` are fixed heuristics, not
  content-aware truncation.** An unusually long or dense paper could have
  its actual conclusion or a late-appearing limitations section cut off
  before the model ever sees it.
- **ROUGE-vs-abstract is a weak, explicitly-labeled sanity check, not a
  real evaluation of critical-analysis quality.** There's no dataset or
  process to actually measure whether the "critical assessment" section
  says anything a domain expert would agree is correct — that would
  require expert human evaluation, which wasn't feasible to build here.
- **No caching of embeddings/analysis across sessions.** `_paper_cache` is
  in-process memory only — a server restart (or Render's free-tier
  spin-down after inactivity) means every paper gets re-downloaded,
  re-extracted, and re-embedded on next use, at real latency and API-call
  cost.
- **Citation-count ranking depends on Semantic Scholar's coverage**, which
  can lag for very recent papers (a paper from the last few weeks may
  genuinely have `citation_count: None` even if it's a real,
  well-regarded piece of work) — the composite ranking doesn't distinguish
  "actually low citations" from "too new to be indexed yet."

---

## 7. Interview-ready summary

*"This tool solves a problem I actually had during my robotics internship
— reading papers efficiently. It has two real capabilities: cross-paper
literature synthesis (search a topic, get the actual shape of the
subfield — approach families, how it's evolved, where approaches trade
off, what's missing — not a list of separate summaries), and single-paper
deep analysis with grounded follow-up Q&A via RAG, so answers come from
the paper's own text rather than the model's general knowledge. The
engineering decisions I'm most able to speak to: everything, including
embeddings, runs through the Gemini API rather than a local model, because
sentence-transformers plus torch was enough to OOM-kill the app on
Render's free 512MB tier — that's a real memory-budget constraint I hit
and fixed, not a default choice. I also had to fix arXiv search
specifically failing on short paper-name acronyms like SPOT and PUMA —
names from my own internship's related work — by adding a title-focused
query that gets merged ahead of a general all-fields search, since a
plain relevance search kept surfacing unrelated papers that happened to be
keyword-dense instead of the actual paper being searched for."*
