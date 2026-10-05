from rag_query import retrieve


def search_finassist_knowledge(query: str) -> dict:
    """
    Search the FinAssist knowledge base and return
    the most relevant policy information with
    traceable policy metadata.
    """

    results = retrieve(
        query,
        top_k=3,
    )

    sources = []

    for result in results:

        sources.append(
            {
                "source": result["source"],
                "policy_id": result["policy_id"],
                "section": result["section"],
                "chunk_id": result["chunk_id"],
                "similarity": round(
                    result["score"],
                    4,
                ),
                "text": result["text"],
            }
        )

    return {
        "query": query,
        "results": sources,
    }


if __name__ == "__main__":

    question = input(
        "Ask FinAssist knowledge question: "
    )

    result = search_finassist_knowledge(
        question
    )

    print("\nRAG TOOL RESULT")
    print("================")

    for item in result["results"]:

        print(
            f"\nSource: {item['source']}"
        )

        print(
            f"Policy ID: {item['policy_id']}"
        )

        print(
            f"Section: {item['section']}"
        )

        print(
            f"Chunk: {item['chunk_id']}"
        )

        print(
            f"Similarity: {item['similarity']}"
        )

        print(
            f"Text:\n{item['text']}"
        )