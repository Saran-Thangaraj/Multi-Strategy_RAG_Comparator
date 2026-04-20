from src.ingestion import load_pdf
from src.chunkers.header_chunker import chunk as header_chunk
from src.chunkers.parent_child_chunker import create_child_chunks, create_parent_chunks
from copy import deepcopy

pages = load_pdf("langchain_rag_technical_docs_clean.pdf")

# Step 1: get header chunks
header_chunks = header_chunk(pages)

# Step 2: deepcopy before passing to parent_child
parent_child_input = deepcopy(header_chunks)

# Step 3: create child and parent
child_chunks = create_child_chunks(parent_child_input)
parent_chunks = create_parent_chunks(child_chunks)

print(f"Child chunks: {len(child_chunks)}")
print(f"Parent chunks: {len(parent_chunks)}")
print(f"\nChild metadata: {child_chunks[0].metadata}")
print(f"\nParent metadata: {parent_chunks[0].metadata}")
