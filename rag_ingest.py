import json
from pathlib import Path

from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.identity import AzureCliCredential, get_bearer_token_provider


load_dotenv()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
VECTOR_STORE = PROJECT_ROOT / "vector_store.json"

EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------

credential = AzureCliCredential()

token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default",
)

client = AzureOpenAI(
    azure_endpoint="https://sureshaspriant-8180-resource.cognitiveservices.azure.com/",
    azure_ad_token_provider=token_provider,
    api_version="2025-01-01-preview",
)


# ---------------------------------------------------------
# Document loading
# ---------------------------------------------------------

def load_documents():
    documents = []

    for file_path in KNOWLEDGE_DIR.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append(
            {
                "source": file_path.name,
                "text": text,
            }
        )

    return documents


# ---------------------------------------------------------
# Section-aware chunking
# ---------------------------------------------------------

def chunk_document(text, source):
    """
    Split a FinAssist policy document into logical sections.

    FinAssist policy documents use uppercase headings such as:

    PURPOSE
    DTI CALCULATION
    DTI BELOW 36%
    INCOME REQUIREMENT
    FINAL DECISION

    Each heading starts a new knowledge chunk.
    """

    lines = text.splitlines()

    chunks = []

    current_section = "General"
    current_lines = []

    # Known FinAssist section headings
    section_headings = {
        "PURPOSE",
        "DTI CALCULATION",
        "DTI BELOW 36%",
        "DTI FROM 36% TO 43%",
        "DTI ABOVE 43%",
        "INCOME REQUIREMENT",
        "DEBT REQUIREMENT",
        "DETERMINISTIC CALCULATION",
        "MISSING INFORMATION",
        "FINAL DECISION",
    }

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        # Check whether this line is a section heading
        if stripped.upper() in section_headings:

            # Save previous section
            if current_lines:

                chunks.append(
                    {
                        "source": source,
                        "section": current_section,
                        "text": "\n".join(current_lines).strip(),
                    }
                )

            # Start new section
            current_section = stripped
            current_lines = []

            continue

        # Keep normal content
        current_lines.append(stripped)

    # Save final section
    if current_lines:

        chunks.append(
            {
                "source": source,
                "section": current_section,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return chunks


# ---------------------------------------------------------
# Extract policy ID
# ---------------------------------------------------------

def extract_policy_id(text):
    """
    Extract the policy ID from the document.
    Handles different capitalization such as:
    Policy ID:
    POLICY ID:
    policy id:
    """

    for line in text.splitlines():

        line = line.strip()

        if line.upper().startswith("POLICY ID:"):
            return line.split(":", 1)[1].strip()

    return "UNKNOWN"


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------

def create_embedding(text):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding


# ---------------------------------------------------------
# Build vector store
# ---------------------------------------------------------

def build_vector_store():

    documents = load_documents()

    print(f"Found {len(documents)} documents.")

    vector_store = []

    for document in documents:

        source = document["source"]
        text = document["text"]

        policy_id = extract_policy_id(text)

        chunks = chunk_document(
            text=text,
            source=source,
        )

        print(
            f"{source}: "
            f"{len(chunks)} sections"
        )

        for index, chunk in enumerate(chunks):

            chunk_text_value = chunk["text"]

            embedding = create_embedding(
                chunk_text_value
            )

            vector_store.append(
                {
                    "source": source,
                    "policy_id": policy_id,
                    "section": chunk["section"],
                    "chunk_id": f"{policy_id}-{index + 1:02d}",
                    "text": chunk_text_value,
                    "embedding": embedding,
                }
            )

    VECTOR_STORE.write_text(
        json.dumps(
            vector_store,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("RAG ingestion completed.")
    print(f"Total chunks: {len(vector_store)}")
    print(f"Vector store: {VECTOR_STORE}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    build_vector_store()