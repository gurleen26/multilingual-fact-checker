from sentence_transformers import SentenceTransformer
import numpy as np

# ── Load model once at startup (not on every call) ──────────
# all-MiniLM-L6-v2 = fast English model, 384 dimensions
# We'll upgrade to multilingual LaBSE in Module 5
model = SentenceTransformer("./models/multilingual")

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
    test_cases = [
        # (query, documents to compare against)
        ("पृथ्वी सपाट है",           # Hindi: "the earth is flat"
         "The earth is not flat. It is an oblate spheroid confirmed by NASA."),

        ("Who walked on the moon?",   # English query
         "नील आर्मस्ट्रांग 1969 में चांद पर चलने वाले पहले इंसान थे।"),  # Hindi answer

        ("La torre Eiffel",           # Spanish
         "The Eiffel Tower was built in 1889 in Paris, France."),
    ]

    print("Cross-lingual similarity scores:")
    print("=" * 55)

    for query, doc in test_cases:
        vecs = embed([query, doc])
        score = cosine_similarity(vecs[0], vecs[1])
        bar = "█" * int(score * 30)
        print(f"\nQuery : {query}")
        print(f"Doc   : {doc[:55]}...")
        print(f"Score : {score:.3f} {bar}")