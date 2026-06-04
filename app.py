import os

from flask import Flask, jsonify, request
from openai import OpenAI
from pinecone import Pinecone

CHUNK_SIZE = 300
OVERLAP_RATIO = 0.2
TOP_K = 7
MAX_CHUNKS_PER_ARTICLE = 3

SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and only "
    "based on the Medium articles dataset context provided to you (metadata and "
    "article passages). You must not use any external knowledge, the open internet, "
    "or information that is not explicitly contained in the retrieved context. "
    "If the answer cannot be determined from the provided context, respond: "
    '"I don\'t know based on the provided Medium articles data." '
    "Always explain your answer using the given context, quoting or paraphrasing "
    "the relevant article passage or metadata when helpful."
)

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL = "4UHRUIN-gpt-5-mini"

app = Flask(__name__)


def _openai():
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def _index():
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
        os.environ["PINECONE_INDEX_NAME"]
    )


def _cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/api/prompt", methods=["POST", "OPTIONS"])
def prompt():
    if request.method == "OPTIONS":
        return _cors(jsonify({}))

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return _cors(jsonify({"error": "question field is required"})), 400

    try:
        client = _openai()
        query_vector = client.embeddings.create(
            model=EMBEDDING_MODEL, input=question
        ).data[0].embedding

        fetch_k = TOP_K * max(3, MAX_CHUNKS_PER_ARTICLE + 1)
        matches = _index().query(
            vector=query_vector, top_k=fetch_k, include_metadata=True
        ).matches

        # Walk candidates in score order, applying per-article cap.
        # If one article dominates all fetch_k results, context may have fewer than TOP_K chunks.
        context = []
        seen: dict[str, int] = {}
        for match in matches:
            if len(context) == TOP_K:
                break
            meta = match.metadata or {}
            aid = str(meta.get("article_id", ""))
            if seen.get(aid, 0) >= MAX_CHUNKS_PER_ARTICLE:
                continue
            seen[aid] = seen.get(aid, 0) + 1
            context.append({
                "article_id": aid,
                "title": str(meta.get("title", "")),
                "chunk": str(meta.get("chunk", "")),
                "score": float(match.score),
            })

        context_text = "\n\n".join(
            f'[{i + 1}] Article: "{c["title"]}"\n{c["chunk"]}'
            for i, c in enumerate(context)
        )
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

        answer = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        ).choices[0].message.content

        return _cors(jsonify({
            "response": answer,
            "context": context,
            "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": user_prompt},
        }))

    except Exception as exc:
        return _cors(jsonify({"error": str(exc)})), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    stats = _index().describe_index_stats()
    return _cors(jsonify({
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K,
        "vector_count": stats.total_vector_count,
        "article_count": 7682,
    }))


