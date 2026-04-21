from src.ingestion import load_pdf
from src.chunkers.header_chunker import chunk
from src.chunkers.parent_child_chunker import create_child_chunks
from src.embeddings import get_embedding_model, store_embeddings

# Setup
embedding = get_embedding_model()

from langchain_chroma import Chroma
child_embedding = Chroma(
    collection_name="child_chunks",
    embedding_function=embedding,
    persist_directory="./chroma_chunks_db"
)

# Test 1: Raw search without filter
print("=== RAW SEARCH ===")
query = "What is Query Decomposition in the context of Advanced RAG Patterns"
results = child_embedding.similarity_search(query, k=5)
for r in results:
    print(r.metadata.get('Header 2'))
    print(r.metadata.get('Header 1'))
    print("----")

# Test 2: Search with noise filter
print("\n=== WITH NOISE FILTER ===")
NOISE_CHAPTERS = {
    "Chapter 14: API Reference",
    "Chapter 2: Python Environment Setup",
    "Appendix E: Testing Your RAG Pipeline",
    "Appendix F: RAG Interview Questions & Model Answers"
}
# How many chunks are indexed?
print("Total chunks:", child_embedding._collection.count())
results2 = child_embedding.similarity_search(
    query, k=5,
    filter={"Header 1": {"$nin": list(NOISE_CHAPTERS)}}
)
for r in results2:
    print(r.metadata.get('Header 2'))
    print(r.metadata.get('Header 1'))
    print("----")