import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.ingestion import load_pdf
from src.embeddings import get_embedding_model, store_embeddings
from src.chunkers import fixed_chunker, header_chunker, parent_child_chunker
from src.retriever import decompose_and_retrieve
from src.reranker import rerank

load_dotenv()

# PDF_PATH = "langchain_rag_technical_docs_clean.pdf"


def build_pipeline(pdf_path):
    print("Loading PDF...")
    pages = load_pdf(pdf_path)

    print("Loading embedding model...")
    embedding = get_embedding_model()

    print("Chunking...")
    fixed_chunks  = fixed_chunker.chunk(pages)
    header_chunks = header_chunker.chunk(pages)
    child_chunks  = parent_child_chunker.create_child_chunks(header_chunks)
    parent_chunks = parent_child_chunker.create_parent_chunks(child_chunks)

    print(f"Fixed chunks  : {len(fixed_chunks)}")
    print(f"Header chunks : {len(header_chunks)}")
    print(f"Child chunks  : {len(child_chunks)}")

    print("Storing embeddings...")
    fixed_embedding  = store_embeddings(fixed_chunks,  "Fixed_Chunks",  embedding)
    header_embedding = store_embeddings(header_chunks, "Header_Chunks", embedding)
    child_embedding  = store_embeddings(child_chunks,  "child_chunks",  embedding)

    print("Storing parent chunks...")
    parent_child_chunker.store_parent_chunks(parent_chunks)

    return (
        embedding,
        fixed_embedding,  fixed_chunks,
        header_embedding, header_chunks,
        child_embedding,  child_chunks
    )


def run_query(query, embedding,
              fixed_embedding,  fixed_chunks,
              header_embedding, header_chunks,
              child_embedding,  child_chunks,
              llm):

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    strategies = {
        "Fixed":        (fixed_embedding,  fixed_chunks),
        "Header":       (header_embedding, header_chunks),
        "Parent-Child": (child_embedding,  child_chunks),
    }

    all_results = {}

    for strategy_name, (retrieval_embedding, chunks) in strategies.items():
        print(f"\n--- Strategy: {strategy_name} ---")

        # Step 1: Retrieve
        unique_context = decompose_and_retrieve(
            query, embedding, chunks, llm, retrieval_embedding
        )

        # Step 2: Rerank
        reranked = rerank(query, unique_context, top_n=5)

        # Step 3: Print results
        print(f"\nTop chunks for [{strategy_name}]:")
        for doc in reranked:
            section = doc.metadata.get('Header 2') or doc.metadata.get('Header 1') or doc.metadata.get('page_chapter') or 'Unknown'
            score = doc.metadata.get('relevance_score', 'N/A')
            print(f"  {section} → {score}")
            # print(f"  Content: {doc.page_content[:150]}")  
            print()


        all_results[strategy_name] = reranked

    return all_results


if __name__ == "__main__":
    (
        embedding,
        fixed_embedding,  fixed_chunks,
        header_embedding, header_chunks,
        child_embedding,  child_chunks
    ) = build_pipeline()

    llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.0)

#     query = """Explain different Chunking Strategies such as:
# - Fixed-Size Chunking
# - Header-Based Chunking
# - Semantic Chunking
# - Parent-Child Chunking"""

    results = run_query(
        #query, 
        embedding,
        fixed_embedding,  fixed_chunks,
        header_embedding, header_chunks,
        child_embedding,  child_chunks,
        llm
    )