# RAG System Design Decisions
## Medium Article RAG Assistant

---

## 1. Dataset

The corpus contains **7,682 English-language Medium articles** with fields: `title`, `text`, `url`, `authors`, `timestamp`, `tags`. Articles average ~6,484 characters (~1,500 tokens) in length.

---

## 2. Chunking Strategy

### Empirical corpus analysis

Before choosing hyperparameters, we measured paragraph-level token lengths across all 230,255 paragraphs in the corpus:

| Statistic | Tokens |
|-----------|--------|
| Mean | 44 |
| Median | 33 |
| p75 | 60 |
| p90 | 91 |
| p95 | 116 |
| p99 | 180 |

66.8% of paragraphs are under 50 tokens. No paragraph exceeds 1,024 tokens.

### Where Medium articles fall in the taxonomy

Medium articles are **long in total length** (~1,500 tokens) but written in a **conversational style** — short, punchy paragraphs typical of online publishing. They do not fit neatly into any single category from the standard taxonomy:

| Category | Chunk size | Overlap | Fits Medium? |
|---|---|---|---|
| General, long articles | 512–1024 | 5–15% | Length yes, style no |
| Conversational data | 200–400 | — | Style yes, length no |

The writing style is the more relevant dimension for chunking: a coherent idea in a Medium article spans 2–4 paragraphs (~70–130 tokens), not 10–15. This places the natural chunk boundary closer to the **conversational** range.

### Chosen hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `chunk_size` | 300 tokens | See rationale below |
| `overlap_ratio` | 0.20 | Slightly above the 5–15% recommendation for long articles, chosen to account for the conversational style where a key sentence on a boundary may carry an idea across paragraphs |
| `top_k` | 5 | General text recommendation is 3–5; upper bound chosen to satisfy the multi-result query type (list 3 articles) while staying within range |
| `max_chunks_per_article` | 2 | With top_k=5 and C=2, worst case is 2+2+1 = 3 distinct articles guaranteed, satisfying the multi-result query requirement |

### Chunk size rationale

The course recommends **512–1024 tokens** for general long-form text as a safe default for most use cases. However, applying that default blindly to this corpus would ignore what the data actually shows.

Measuring paragraph lengths across all 230,255 paragraphs in the corpus:

- **Mean paragraph length: 44 tokens**
- **Median paragraph length: 33 tokens**
- **99th percentile: 180 tokens**

A 512-token chunk therefore spans roughly **10–15 paragraphs** — and Medium articles do not have 15-paragraph sections. They have short, focused paragraphs, each making one point. Grouping 15 of them into a single embedding vector dilutes the signal: a query about one specific idea retrieves a chunk that contains a dozen other ideas, and cosine similarity degrades accordingly.

We chose **300 tokens** as a principled middle ground: large enough to capture 3–5 consecutive paragraphs (a natural section in a Medium article) and provide sufficient context for the language model to reason from, but small enough that each chunk remains semantically focused. This keeps the embedding tight around a single coherent idea rather than a blend of several.

### Over-fetching for diversity

Pinecone is queried with `fetch_k = top_k × 3 = 15` candidates. Results are then filtered greedily: chunks are accepted in score order, skipping any article that has already contributed `max_chunks_per_article = 2` chunks. The final context contains exactly `top_k = 5` chunks, drawn from at least 3 distinct articles.

---

## 3. Embedding Model

**Model:** `4UHRUIN-text-embedding-3-small`
**Dimensions:** 1,536
**Total chunks:** ~28,470
**Estimated embedding cost:** ~$0.29 (well within the $5 budget)

---

## 4. Generation Model

**Model:** `4UHRUIN-gpt-5-mini`

### System prompt (required, verbatim)

> You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don't know based on the provided Medium articles data." Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.

---

## 5. Vector Database

**Provider:** Pinecone (Serverless, AWS us-east-1)
**Metric:** Cosine similarity
**Dimensions:** 1,536

---

## 6. Deployment

**Platform:** Vercel (Python serverless runtime)

Streamlit was considered but ruled out: it runs a stateful WebSocket server and cannot expose the REST endpoints (`POST /api/prompt`, `GET /api/stats`) required by the assignment. Vercel's Python serverless runtime was chosen instead — each `api/*.py` file becomes a standalone HTTP function, exposing exactly the required endpoints with no Node.js dependency. A plain HTML demo page at `/` provides an interactive front-end.

---

## 7. API Endpoints

### `POST /api/prompt`

Input:
```json
{ "question": "Your natural language question here" }
```

Output:
```json
{
  "response": "Final answer from the model.",
  "context": [
    { "article_id": "123", "title": "...", "chunk": "...", "score": 0.87 }
  ],
  "Augmented_prompt": {
    "System": "...",
    "User": "..."
  }
}
```

### `GET /api/stats`

```json
{ "chunk_size": 300, "overlap_ratio": 0.2, "top_k": 5 }
```
