"""Local end-to-end test: runs 5 example queries against the live Pinecone index.

Usage:
    # Fill in .env with your credentials, then:
    python scripts/test_local.py
"""

import json
import os

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from pinecone import Pinecone

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL = "4UHRUIN-gpt-5-mini"
CHUNK_SIZE = 300
TOP_K = 8
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

QUESTIONS = [
    # 1. Precise fact retrieval
    "Find an article that reframes marketing as a conversation with readers, aimed at writers who find self-promotion uncomfortable. Provide the title and author.",
    # 2. Multi-result topic listing
    "List exactly 3 articles about education. Return only the titles.",
    # 3. Key idea summary extraction
    "Find an article that argues past pandemics (such as the bubonic plague) can spur innovation and recovery, and summarise its central argument.",
    # 4. Recommendation with evidence-based justification
    "I want practical, beginner-friendly advice on building habits that actually stick. Which article would you recommend, and why?",
    # 5. Bonus: metadata-based retrieval
    "Find an article about mental health and self-compassion. What is its main message?",
]


def ask(client: OpenAI, index, question: str) -> dict:
    emb = client.embeddings.create(model=EMBEDDING_MODEL, input=question)
    query_vector = emb.data[0].embedding

    fetch_k = TOP_K * max(3, MAX_CHUNKS_PER_ARTICLE + 1)
    results = index.query(vector=query_vector, top_k=fetch_k, include_metadata=True)

    context = []
    chunks_per_article: dict[str, int] = {}
    for match in results.matches:
        if len(context) == TOP_K:
            break
        meta = match.metadata or {}
        article_id = str(meta.get("article_id", ""))
        if chunks_per_article.get(article_id, 0) >= MAX_CHUNKS_PER_ARTICLE:
            continue
        chunks_per_article[article_id] = chunks_per_article.get(article_id, 0) + 1
        context.append({
            "article_id": article_id,
            "title": str(meta.get("title", "")),
            "chunk": str(meta.get("chunk", "")),
            "score": float(match.score),
        })

    context_text = "\n\n".join(
        f'[{i + 1}] Article: "{c["title"]}"\n{c["chunk"]}'
        for i, c in enumerate(context)
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return {
        "response": completion.choices[0].message.content,
        "context": context,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        },
    }


def main():
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    stats = index.describe_index_stats()
    print(f"Pinecone index has {stats.total_vector_count:,} vectors\n")
    print("=" * 70)

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\nQ{i}: {question}")
        print("-" * 70)
        result = ask(client, index, question)

        print(f"RESPONSE:\n{result['response']}")
        print(f"\nCONTEXT ({len(result['context'])} chunks retrieved):")
        for c in result["context"]:
            print(f"  [{c['article_id']}] \"{c['title']}\"  score={c['score']:.4f}")

        print("=" * 70)


if __name__ == "__main__":
    main()
