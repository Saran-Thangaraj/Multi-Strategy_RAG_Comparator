# Multi-Strategy RAG Pipeline

A production-grade Retrieval-Augmented Generation system that compares three chunking strategies — Fixed-Size, Header-Based, and Parent-Child — on the same query, with hybrid retrieval, query decomposition, and reranking.

---

## Project Structure

```
rag_project/
    main.py                          # Wires everything together
    streamlit_app.py                 # UI for strategy comparison
    requirements.txt
    src/
        ingestion.py                 # PDF loading
        embeddings.py                # Hash-based deduplication + ChromaDB
        retriever.py                 # Query decomposition + hybrid retrieval
        reranker.py                  # Cohere reranking with threshold filter
        chunkers/
            fixed_chunker.py         # RecursiveCharacterTextSplitter
            header_chunker.py        # MarkdownHeaderTextSplitter
            parent_child_chunker.py  # Parent + Child chunks
```

---

## Pipeline Flow

```
PDF Upload
    ↓
3 Chunking Strategies (Fixed, Header, Parent-Child)
    ↓
Embed + Store in ChromaDB (with deduplication)
    ↓
User Query
    ↓
is_multi_topic? → Query Decomposition (Retrieval-Augmented)
    ↓
Hybrid Retrieval (BM25 + Vector) per sub-query
    ↓
Cohere Reranking
    ↓
Compare results across 3 strategies
```

---

## Engineering Decisions & Challenges

### Challenge 1: Page Boundary Bleeding

**Problem:** Processing pages independently caused content from the end of one page to overflow into the first chunk of the next page — resulting in incorrect header assignments. For example, code from section 6.2 appeared under the 6.3 header chunk.

**Decision:** Moved from page-by-page processing to full document processing — merging all pages into a single text before splitting. This ensures `MarkdownHeaderTextSplitter` sees the complete document structure and creates clean boundaries based on real headers rather than arbitrary page breaks.

```
Page by page = good metadata, bad boundaries
Full document = good boundaries, correct metadata
```

---

### Challenge 2: Fake Headers Inside Code Examples

**Problem:** Code examples containing multiline strings (e.g., `SAMPLE_MD = """`) included markdown-style headers like `# Chapter 1` inside them. The splitter incorrectly treated these as real document headers, creating duplicate chunks that contaminated retrieval results.

**Decision:** Added custom `MarkdownHeaderTextSplitter` with code block detection to skip headers found inside code examples.

---

### Parent-Child: parent_id Design Decision

The `parent_id` is constructed by combining three fields — `source + Header1 + Header2` — rather than a sequential number or random UUID.

**Why:** Each identifier becomes self-describing. Anyone reading the metadata immediately understands which document, chapter, and section a chunk belongs to — without querying the database.

```
langchain_rag_technical_docs_clean.pdf-Chapter 4: Chunking Strategies-4.1 Fixed-Size Chunking-12
```

---

### Hash-Based Deduplication

**Content-Aware Uniqueness** — Each chunk is fingerprinted by its actual content + metadata combined. The same file uploaded with a different name is still correctly identified as a duplicate.

**Performance at Scale** — ChromaDB is queried directly by hash ID, making deduplication O(k) where k is only the new chunks being uploaded — not the entire database.

**Chunk-Level Precision** — If a document is updated and only 3 pages change, only those 3 new chunks get re-indexed. Unchanged chunks are safely skipped.

---

### Retrieval Strategy Decision

**Problem:** Irrelevant chunks (2.4, 14.2, E.1) were scoring higher than relevant chunks (4.2) because:
- Similarity search only measures vector closeness
- Chunks containing the word "chunker" in code scored high regardless of actual topic relevance

**Why I rejected MMR:**
MMR fixes diversity between retrieved chunks. It does NOT fix why irrelevant chunks score high in the first place. MMR would still select 2.4 Project Structure as Rank 1 and diversify FROM it — not remove it.

**Why Reranking wins:**
- Takes ALL retrieved chunks
- Scores Query + Chunk TOGETHER using cross-encoder
- Understands full context relationship
- 2.4 Project Structure → low score (just file paths)
- 4.2 Header-Based → high score (actual explanation)

---

### Bug: Reranker Giving Low Scores to Correct Chunks

**Query:**
```
- Similarity Search vs MMR
- Hybrid Search
- Reranking
```

**Expected:** 7.1, 7.2, 7.3 should rank high after reranking.

**What actually happened:** F.3 ranked higher than 7.1, 7.2, 7.3.

**Root Cause:** Similarity search converts the entire multi-topic query into a single vector representing the average meaning of all 3 topics. So 7.1 only matched one-third of that vector. F.3 matched more because it broadly touched all words. Reranker also saw the full query vs each chunk — so 7.1 still only satisfied one-third of the query.

**Fix:** Query Decomposition — break the query into separate questions, retrieve for each, then deduplicate results.

---

### Problem: LLM Generates Hallucinated Sub-queries

**Problem:**
```
User asks multi-topic query
→ LLM generates sub-queries from imagination
→ Sub-queries don't match document content
→ Wrong chunks retrieved
```

**Why it happened:** LLM had no knowledge of what sections exist in the document → Generated generic internet-style queries → "parsing algorithms for files" (not in document).

**Fix — Retrieval-Augmented Query Decomposition:**

```
Step 1: Hybrid search on section names
        (BM25 catches exact keywords + Vector catches semantic meaning)
        → Finds relevant sections from YOUR document

Step 2: Pass those sections to LLM
        → LLM now knows what actually exists
        → Generates grounded sub-queries

Step 3: Sub-queries match real document content
        → Correct chunks retrieved
```

---

## Tech Stack

- **LangChain** — document loading, text splitting, retrieval chains
- **ChromaDB** — vector store with persistent storage
- **HuggingFace** — sentence-transformers embeddings
- **Cohere** — reranking with cross-encoder
- **Groq (LLaMA 3.3)** — query decomposition LLM
- **BM25** — keyword retrieval for hybrid search
- **Streamlit** — UI for strategy comparison

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

Add your API keys in the sidebar:
- Groq API Key
- Cohere API Key

Upload any PDF and compare chunking strategies side by side.


## Query Rewriting for Single-Topic Queries

### Problem
Short generic queries like "what is Query Decomposition?" 
embed poorly — similarity search returns wrong chunks 
(Chapter 1 instead of Chapter 11).

### Root Cause
Embedding model maps "what is X" to introductory content 
regardless of the actual topic X.

### Fix — HyDE-Style Query Rewriting
1. Hybrid search on section headers → finds relevant section
   "what is Query Decomposition?" → finds 11.2 Query Decomposition
   
2. LLM rewrites query using section as context:
   "what is Query Decomposition?" 
   → "What are query decomposition techniques for breaking 
      down complex queries into subqueries in RAG patterns?"

3. Rewritten query embeds correctly → retrieves right chunk
   Score: 0.997 ✅

### Why This Works
Section headers act as grounding context for LLM.
LLM generates domain-specific query instead of generic one.
Strong query → strong embedding → correct retrieval.

### Inspired By
HyDE (Hypothetical Document Embeddings) — using LLM to 
bridge gap between vague query and specific document content.
