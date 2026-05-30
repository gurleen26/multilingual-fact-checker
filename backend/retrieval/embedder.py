from sentence_transformers import SentenceTransformer
import numpy as np

# ── Load model once at startup (not on every call) ──────────
# all-MiniLM-L6-v2 = fast English model, 384 dimensions
# We'll upgrade to multilingual LaBSE in Module 5
model = SentenceTransformer("./models/all-MiniLM-L6-v2")

def embed(texts: list[str]) -> np.ndarray:
    """Convert a list of strings into a 2D array of vectors."""
    return model.encode(texts, normalize_embeddings=True)
    # normalize_embeddings=True means we can use dot product
    # instead of full cosine similarity — same result, faster

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """How similar are two vectors? Returns 0.0 to 1.0."""
    return float(np.dot(vec_a, vec_b))  # works because vectors are normalized

# ── Test it ─────────────────────────────────────────────────
if __name__ == "__main__":
    sentences = [
        "Who built the Eiffel Tower?",           # query
        "The Eiffel Tower was constructed in 1889 by Gustave Eiffel.",  # relevant
        "Python is a programming language.",      # not relevant
        "The iron structure in Paris was designed by Eiffel.",  # relevant (no shared words!)
    ]

    vectors = embed(sentences)

    query_vec = vectors[0]
    print(f"Embedding shape: {vectors.shape}")  # (4, 384)
    print(f"\nQuery: '{sentences[0]}'")
    print(f"\nSimilarity scores:")

    for i in range(1, len(sentences)):
        score = cosine_similarity(query_vec, vectors[i])
        bar = "█" * int(score * 20)
        print(f"  {score:.3f} {bar} → '{sentences[i][:50]}'")