import json
from pathlib import Path

import numpy as np
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv


load_dotenv()

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
VECTOR_STORE = PROJECT_ROOT / "vector_store.json"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"

AZURE_ENDPOINT = (
    "https://sureshaspriant-8180-resource.cognitiveservices.azure.com/"
)


# ---------------------------------------------------------
# Azure client
# ---------------------------------------------------------

credential = AzureCliCredential()

token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default",
)

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    azure_ad_token_provider=token_provider,
    api_version="2025-01-01-preview",
)


# ---------------------------------------------------------
# Load vector store
# ---------------------------------------------------------

def load_vector_store():
    with open(VECTOR_STORE, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# Create query embedding
# ---------------------------------------------------------

def create_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return np.array(response.data[0].embedding)


# ---------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------

def cosine_similarity(vector_a, vector_b):

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0

    return np.dot(vector_a, vector_b) / denominator


# ---------------------------------------------------------
# Retrieve relevant chunks
# ---------------------------------------------------------

def retrieve(query, top_k=3):

    vector_store = load_vector_store()

    query_embedding = create_embedding(query)

    query_lower = query.lower()

    # ---------------------------------------------------------
    # Identify important policy terms
    # ---------------------------------------------------------

    keyword_scores = {}

    if "above 43" in query_lower or "over 43" in query_lower:
        keyword_scores["DTI ABOVE 43%"] = 0.25

    if "36" in query_lower and "43" in query_lower:
        keyword_scores["DTI FROM 36% TO 43%"] = 0.25

    if "below 36" in query_lower or "under 36" in query_lower:
        keyword_scores["DTI BELOW 36%"] = 0.25

    if "income" in query_lower:
        keyword_scores["INCOME REQUIREMENT"] = 0.15

    if "debt" in query_lower:
        keyword_scores["DEBT REQUIREMENT"] = 0.15

    if "deterministic" in query_lower:
        keyword_scores["DETERMINISTIC CALCULATION"] = 0.15

    if "missing" in query_lower:
        keyword_scores["MISSING INFORMATION"] = 0.15

    results = []

    for item in vector_store:

        document_embedding = np.array(
            item["embedding"]
        )

        semantic_score = cosine_similarity(
            query_embedding,
            document_embedding,
        )

        section = item.get(
            "section",
            "General",
        )

        # -----------------------------------------------------
        # Keyword / section boost
        # -----------------------------------------------------

        keyword_boost = 0.0

        for keyword, boost in keyword_scores.items():

            if keyword.lower() == section.lower():
                keyword_boost = max(
                    keyword_boost,
                    boost,
                )

        # -----------------------------------------------------
        # Exact phrase matching
        # -----------------------------------------------------

        text_lower = item["text"].lower()

        exact_match_boost = 0.0

        if "above 43" in query_lower:
            if "above 43%" in text_lower:
                exact_match_boost = 0.20

        if "below 36" in query_lower:
            if "below 36%" in text_lower:
                exact_match_boost = 0.20

        if "36" in query_lower and "43" in query_lower:
            if "36% through 43%" in text_lower:
                exact_match_boost = 0.20

        final_score = (
            semantic_score
            + keyword_boost
            + exact_match_boost
        )

        results.append(
            {
                "source": item["source"],
                "policy_id": item.get(
                    "policy_id",
                    "UNKNOWN",
                ),
                "section": section,
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "score": float(semantic_score),
                "keyword_boost": float(keyword_boost),
                "exact_match_boost": float(
                    exact_match_boost
                ),
                "final_score": float(final_score),
            }
        )

    # ---------------------------------------------------------
    # Sort using reranked score
    # ---------------------------------------------------------

    results.sort(
        key=lambda x: x["final_score"],
        reverse=True,
    )

    return results[:top_k]


# ---------------------------------------------------------
# Generate grounded answer
# ---------------------------------------------------------

def generate_answer(query, retrieved_chunks):

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"""
SOURCE: {chunk['source']}
CHUNK: {chunk['chunk_id']}
CONTENT:
{chunk['text']}
"""
        )

    context = "\n".join(context_parts)

    system_prompt = """
You are FinAssist AI, an enterprise financial document
analysis assistant.

Answer the user's question using ONLY the supplied
FinAssist policy context.

Rules:

1. Do not invent policy information.
2. If the answer is not present in the context, say so.
3. Cite the source document used.
4. Clearly distinguish policy information from assumptions.
5. Do not make financial approval decisions.
6. Do not fabricate applicant information.
"""

    user_prompt = f"""
FINASSIST KNOWLEDGE CONTEXT
---------------------------

{context}

---------------------------

USER QUESTION

{query}

Provide a concise answer and include the source document.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    question = input(
        "\nAsk FinAssist a policy question: "
    )

    print("\nSearching knowledge base...\n")

    results = retrieve(question)

    print("RETRIEVED CHUNKS")
    print("================")

    for result in results:

        print(
            f"\nSource: {result['source']}"
        )
        
        print(
            f"Policy ID: {result['policy_id']}"
        )

        print(
            f"Section: {result['section']}"
        )

        print(
            f"Chunk: {result['chunk_id']}"
        )

        print(
            f"Semantic similarity: {result['score']:.4f}"
        )

        print(
            f"Keyword boost: {result['keyword_boost']:.4f}"
        )

        print(
            f"Exact match boost: {result['exact_match_boost']:.4f}"
        )

        print(
            f"Final score: {result['final_score']:.4f}"
        )

        print(
            f"Text:\n{result['text']}"
        )

    print("\nGenerating grounded answer...\n")

    answer = generate_answer(
        question,
        results,
    )

    print("FINASSIST ANSWER")
    print("================")
    print(answer)