from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter,
    FieldCondition, MatchValue, Range
)
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

COLLECTION = os.getenv("COLLECTION_NAME", "fact_checker_docs")
VECTOR_SIZE = 384   # must match all-MiniLM-L6-v2 output

# ── Connect ──────────────────────────────────────────────────
# :memory: = no Docker needed, data lives in RAM during the session
# swap to QdrantClient(url="http://localhost:6333") when you install Docker
client = QdrantClient(":memory:")

def create_collection():
    """Create the collection if it doesn't exist yet."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"Created collection: {COLLECTION}")
    else:
        print(f"Collection already exists: {COLLECTION}")

def upsert_documents(texts: list[str], vectors: list, metadatas: list[dict]):
    """
    Store documents with their vectors and metadata.
    texts     = list of raw text chunks
    vectors   = list of embeddings (from embedder.py)
    metadatas = list of dicts with source, language, credibility etc.
    """
    points = []
    for i, (text, vector, meta) in enumerate(zip(texts, vectors, metadatas)):
        payload = {"text": text, **meta}   # merge text + metadata into payload
        points.append(PointStruct(
            id=str(uuid.uuid4()),          # unique ID for each point
            vector=vector.tolist(),        # numpy array → plain list
            payload=payload
        ))

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Stored {len(points)} points in Qdrant")

def search(query_vector, top_k: int = 5, lang_filter: str = None):
    """
    Search for similar documents.
    Optionally filter by language (e.g. lang_filter="en")
    """
    search_filter = None
    if lang_filter:
        search_filter = Filter(must=[
            FieldCondition(
                key="language",
                match=MatchValue(value=lang_filter)
            )
        ])

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector.tolist(),
        query_filter=search_filter,
        limit=top_k
    ).points
    return results

# ── Test everything end to end ───────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from backend.retrieval.embedder import embed

    create_collection()

    # Sample documents with metadata
    docs = [
        ("The Eiffel Tower was built in 1889 by Gustave Eiffel.",
         {"source": "britannica.com", "language": "en", "credibility": 0.95}),
        ("Neil Armstrong was the first human to walk on the moon in 1969.",
         {"source": "nasa.gov",       "language": "en", "credibility": 0.99}),
        ("Python was created by Guido van Rossum and first released in 1991.",
         {"source": "python.org",     "language": "en", "credibility": 0.98}),
        ("La Torre Eiffel fue construida en 1889.",
         {"source": "wikipedia.org",  "language": "es", "credibility": 0.90}),
    ]

    texts    = [d[0] for d in docs]
    metas    = [d[1] for d in docs]
    vectors  = embed(texts)

    upsert_documents(texts, vectors, metas)

    # Search 1 — no filter
    print("\n── Search: 'iron tower in France' (no filter) ──")
    query_vec = embed(["iron tower in France"])[0]
    results = search(query_vec, top_k=2)
    for r in results:
        print(f"  {r.score:.3f} | {r.payload['source']} | {r.payload['text'][:60]}")

    # Search 2 — English only filter
    print("\n── Search: 'Eiffel Tower' (English only) ──")
    query_vec = embed(["Eiffel Tower"])[0]
    results = search(query_vec, top_k=3, lang_filter="en")
    for r in results:
        print(f"  {r.score:.3f} | lang={r.payload['language']} | {r.payload['text'][:55]}")