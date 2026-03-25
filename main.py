"""
GarageAI — RAG-based vehicle maintenance assistant for the 2018 Audi A4 (B9).

Usage:
    python main.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient

load_dotenv()

# ---------- Configuration ----------

COLLECTION_NAME = "garageai_2018_audi_a4"
QDRANT_PATH = ".qdrant_data"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 6

# ---------- Langfuse (optional) ----------

langfuse_handler = None
try:
    from langfuse.callback import CallbackHandler
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        langfuse_handler = CallbackHandler()
        print("[Langfuse] Tracing enabled")
    else:
        print("[Langfuse] No keys found — tracing disabled")
except ImportError:
    print("[Langfuse] Not installed — tracing disabled")

# ---------- System Prompt ----------

SYSTEM_PROMPT = """\
You are GarageAI, an expert vehicle maintenance assistant specializing in the 2018 Audi A4 (B9 platform, 2.0T TFSI engine, 7-speed S tronic transmission).

Your job is to help owners make informed maintenance decisions by providing accurate, \
sourced guidance. You draw from three types of knowledge:

1. **OEM Documentation** — Audi's official owner's manual and maintenance schedules. \
These are the authoritative baseline for service intervals and fluid specs.
2. **NHTSA Data** — Government complaints, recalls, and investigations. These reveal \
real-world failure patterns that go beyond what the manual covers.
3. **Community Knowledge** — Enthusiast forums (AudiWorld, Audizine) where experienced \
owners share what actually happens at various mileage points.

When answering:
- Always cite which source type your information comes from (OEM, NHTSA, community).
- If OEM guidance and community experience differ (e.g., oil change intervals), present both perspectives and explain the tradeoff.
- When asked "what should I do at X miles," prioritize by: safety-critical items first, then manufacturer-required, then community-recommended preventive items.
- Be direct about costs when you have data. Owners want to budget.
- If you're uncertain or the data doesn't cover the question, say so clearly rather than guessing.
- Keep responses practical and actionable — this is a tool for real car owners, not a textbook.

IMPORTANT: You are focused on the 2018 Audi A4 specifically. If asked about other vehicles, \
clarify that your knowledge base is scoped to this model and may not apply.
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Based on the following retrieved context, answer the user's question.

--- RETRIEVED CONTEXT ---
{context}
--- END CONTEXT ---

User question: {question}

Provide a helpful, sourced answer. Cite whether info comes from OEM docs, NHTSA data, or community knowledge."""),
])

# ---------- Retriever ----------


def get_retriever():
    """Build a retrieval function using Qdrant."""
    client = QdrantClient(path=QDRANT_PATH)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    def retrieve(query: str) -> str:
        query_vector = embeddings.embed_query(query)
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K,
        ).points

        if not results:
            return "No relevant documents found in the knowledge base."

        context_parts = []
        for r in results:
            source = r.payload.get("source_type", "unknown")
            filename = r.payload.get("filename", "unknown")
            text = r.payload.get("text", "")
            context_parts.append(
                f"[Source: {source} | File: {filename}]\n{text}"
            )

        return "\n\n---\n\n".join(context_parts)

    return retrieve


# ---------- Chain ----------


def build_chain():
    """Build the RAG chain."""
    retrieve = get_retriever()
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)

    chain = (
        {
            "context": lambda x: retrieve(x["question"]),
            "question": lambda x: x["question"],
        }
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    return chain


# ---------- CLI ----------


def main():
    print("=" * 60)
    print("  GarageAI — 2018 Audi A4 Maintenance Assistant")
    print("=" * 60)
    print("Ask me anything about maintaining your 2018 A4.")
    print("Type 'quit' to exit.\n")

    chain = build_chain()

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nGarageAI: ", end="", flush=True)
        try:
            config = {}
            if langfuse_handler:
                config["callbacks"] = [langfuse_handler]

            response = chain.invoke(
                {"question": question},
                config=config,
            )
            print(response)
        except Exception as e:
            print(f"\nError: {e}")

        print()


if __name__ == "__main__":
    main()
