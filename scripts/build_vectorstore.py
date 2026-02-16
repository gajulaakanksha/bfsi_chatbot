"""Build ChromaDB vector store from knowledge base Markdown documents.

Uses LangChain document loaders, text splitters, and HuggingFace embeddings
to create a persistent vector store for the RAG pipeline.
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

KNOWLEDGE_BASE_DIR = os.path.join("data", "knowledge_base")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "bfsi_knowledge"


def load_documents():
    """Load all Markdown documents from the knowledge base directory."""
    loader = DirectoryLoader(
        KNOWLEDGE_BASE_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {KNOWLEDGE_BASE_DIR}")
    return docs


def split_documents(docs):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (chunk_size=800, overlap=150)")
    return chunks


def build_vectorstore(chunks):
    """Create ChromaDB vector store from document chunks."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # Use CPU for embeddings (small model)
        encode_kwargs={"normalize_embeddings": True},
    )

    # Remove old vector store if it exists
    if os.path.exists(CHROMA_PERSIST_DIR):
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print(f"Removed existing vector store at {CHROMA_PERSIST_DIR}")

    print("Building ChromaDB vector store (this may take a minute)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # Quick sanity check
    count = vectorstore._collection.count()
    print(f"Vector store built successfully with {count} vectors")
    print(f"Persisted to: {CHROMA_PERSIST_DIR}")

    # Test retrieval
    print("\n--- Sanity Check: Test Query ---")
    test_query = "What is the eligibility criteria for a home loan?"
    results = vectorstore.similarity_search_with_score(test_query, k=3)
    for i, (doc, score) in enumerate(results):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        print(f"  [{i+1}] score={score:.4f} | source={source}")
        print(f"      {doc.page_content[:120]}...")
    print("--- Done ---")

    return vectorstore


def main():
    print("=" * 60)
    print("BFSI Knowledge Base → ChromaDB Vector Store Builder")
    print("=" * 60)

    docs = load_documents()
    if not docs:
        print("ERROR: No documents found. Check the knowledge_base directory.")
        sys.exit(1)

    chunks = split_documents(docs)
    build_vectorstore(chunks)

    print("\n✓ Vector store is ready for RAG retrieval.")


if __name__ == "__main__":
    main()
