import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Step 1: Our "database" — just a list of strings for now ──
documents = [
    "The Eiffel Tower is located in Paris, France. It was built in 1889.",
    "The Great Wall of China stretches over 13,000 miles.",
    "The moon landing occurred on July 20, 1969. Neil Armstrong was first.",
    "Python was created by Guido van Rossum and released in 1991.",
    "The Amazon River is the largest river by discharge in the world.",
]

# ── Step 2: Retrieve — find the most relevant document ──
def retrieve(question: str, docs: list) -> str:
    """Toy retrieval: find the doc with the most word overlap."""
    question_words = set(question.lower().split())
    best_doc = max(docs, key=lambda d: len(set(d.lower().split()) & question_words))
    return best_doc

# ── Step 3: Generate — answer using only the retrieved doc ──
def rag_answer(question: str) -> str:
    context = retrieve(question, documents)
    print(f"\n📄 Retrieved: {context}")   # so you can see what was found

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=200,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the document below. "
                    "If the document doesn't contain the answer, say 'Not in my knowledge base.'\n\n"
                    f"Document:\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content

# ── Test it ──
if __name__ == "__main__":
    questions = [
        "When was the Eiffel Tower built?",
        "Who created Python?",
        "What is the capital of Japan?",   # not in our docs — watch what happens
    ]
    for q in questions:
        print(f"\n❓ Question: {q}")
        print(f"💬 Answer: {rag_answer(q)}")