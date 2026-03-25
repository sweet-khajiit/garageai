"""
Process collected data: load documents, chunk them, embed, and store in Qdrant.

Usage:
    python scripts/process_data.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
COLLECTION_NAME = "garageai_2018_audi_a4"
QDRANT_PATH = Path(__file__).parent.parent / ".qdrant_data"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def load_pdfs() -> list[Document]:
    """Load all PDFs from data/raw/."""
    docs = []
    for pdf_path in RAW_DIR.glob("*.pdf"):
        print(f"  Loading PDF: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pdf_docs = loader.load()
        # Tag each page with source metadata
        for doc in pdf_docs:
            doc.metadata["source_type"] = "oem_document"
            doc.metadata["filename"] = pdf_path.name
        docs.extend(pdf_docs)
    return docs


def load_text_files() -> list[Document]:
    """Load all .txt files from data/raw/."""
    docs = []
    for txt_path in RAW_DIR.glob("*.txt"):
        print(f"  Loading text: {txt_path.name}")

        # NHTSA files get split on the --- separator
        if "nhtsa" in txt_path.name.lower():
            content = txt_path.read_text()
            entries = content.split("\n---\n\n")
            source_type = "nhtsa_data"
            for entry in entries:
                if entry.strip():
                    docs.append(Document(
                        page_content=entry.strip(),
                        metadata={
                            "source_type": source_type,
                            "filename": txt_path.name,
                        }
                    ))
        else:
            # Forum posts, articles, etc.
            loader = TextLoader(str(txt_path))
            text_docs = loader.load()
            for doc in text_docs:
                doc.metadata["source_type"] = "community_knowledge"
                doc.metadata["filename"] = txt_path.name
            docs.extend(text_docs)

    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into chunks suitable for RAG retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Don't re-chunk NHTSA entries — they're already individual records
    nhtsa_docs = [d for d in docs if d.metadata.get("source_type") == "nhtsa_data"]
    other_docs = [d for d in docs if d.metadata.get("source_type") != "nhtsa_data"]

    chunked = splitter.split_documents(other_docs)
    chunked.extend(nhtsa_docs)

    print(f"  Total chunks: {len(chunked)} ({len(nhtsa_docs)} NHTSA + {len(chunked) - len(nhtsa_docs)} other)")
    return chunked


def store_in_qdrant(chunks: list[Document]):
    """Embed chunks and store in a local Qdrant collection."""
    print(f"\n  Initializing Qdrant at {QDRANT_PATH}...")
    client = QdrantClient(path=str(QDRANT_PATH))

    # Recreate collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    print(f"  Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Batch embed
    batch_size = 50
    all_points = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.page_content for c in batch]
        vectors = embeddings.embed_documents(texts)

        for j, (chunk, vector) in enumerate(zip(batch, vectors)):
            point = PointStruct(
                id=i + j,
                vector=vector,
                payload={
                    "text": chunk.page_content,
                    "source_type": chunk.metadata.get("source_type", "unknown"),
                    "filename": chunk.metadata.get("filename", "unknown"),
                    "page": chunk.metadata.get("page", None),
                },
            )
            all_points.append(point)

        print(f"    Embedded batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")

    client.upsert(collection_name=COLLECTION_NAME, points=all_points)
    print(f"\n  Stored {len(all_points)} points in collection '{COLLECTION_NAME}'")


def main():
    print("=" * 60)
    print("GarageAI — Data Processing Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading PDFs...")
    pdf_docs = load_pdfs()
    print(f"  Loaded {len(pdf_docs)} pages from PDFs")

    print("\n[2/4] Loading text files...")
    text_docs = load_text_files()
    print(f"  Loaded {len(text_docs)} text documents")

    all_docs = pdf_docs + text_docs
    if not all_docs:
        print("\n  ERROR: No documents found in data/raw/")
        print("  Run collect_nhtsa.py first, then add PDFs and text files.")
        return

    print(f"\n[3/4] Chunking {len(all_docs)} documents...")
    chunks = chunk_documents(all_docs)

    print("\n[4/4] Embedding and storing in Qdrant...")
    store_in_qdrant(chunks)

    print("\n" + "=" * 60)
    print("Done! Your vector store is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
