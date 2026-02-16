"""Tier 3 – RAG Engine.

Retrieves relevant chunks from the ChromaDB vector store and uses the
language model to generate a grounded response based on the retrieved
context.  Uses LangChain components for retrieval and chain building.
"""
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "bfsi_knowledge"
RAG_K = 3  # number of chunks to retrieve


class RAGEngine:
    """Retrieve relevant knowledge base chunks via ChromaDB."""

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RAG_K},
        )

    def retrieve(self, query: str, k: int = RAG_K):
        """Return list of (content, metadata, score) tuples."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "score": float(score),
            }
            for doc, score in results
        ]

    def get_context_string(self, query: str, k: int = RAG_K) -> str:
        """Return a formatted context string for the LLM prompt."""
        chunks = self.retrieve(query, k=k)
        if not chunks:
            return ""
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[Source: {c['source']}]\n{c['content']}")
        return "\n\n---\n\n".join(parts)
