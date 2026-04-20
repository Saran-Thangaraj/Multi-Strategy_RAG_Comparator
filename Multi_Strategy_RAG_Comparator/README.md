 Challenge 1: Page Boundary Bleeding
During development, I discovered that processing pages independently caused a critical boundary issue — content from the end of one page would overflow into the first chunk of the next page, resulting in chunks with incorrect header assignments. For example, code examples from section 6.2 were incorrectly appearing under the 6.3 header chunk.
Decision: Full Document Chunking
To solve this, I moved from page-by-page processing to full document processing — merging all pages into a single text before splitting. This ensures that MarkdownHeaderTextSplitter sees the complete document structure and creates clean, accurate boundaries based on real headers and sub-headers rather than arbitrary page breaks.
Challenge 2: Fake Headers Inside Code Examples
While testing with real technical documentation, I discovered that code examples containing multiline strings (e.g., SAMPLE_MD = """) included markdown-style headers like # Chapter 1 inside them. The splitter incorrectly treated these as real document headers, creating duplicate chunks that contaminated retrieval results.
Decision: Code Block Detection
I have added the custom markdownheadersplitter and it works as expected 


Page by page = good metadata, bad boundaries
Full document = good boundaries, hard metadata 

Parent_child:

Parent ID Design Decision
In this project, the parent_id is constructed by combining three meaningful fields — source, Header1, and Header2 — rather than using a simple sequential number or random UUID. This decision was driven by the nature of technical documentation, where every section has a clear hierarchical structure: a document source, a chapter heading, and a sub-section heading. By encoding this hierarchy directly into the parent_id, each identifier becomes self-describing, meaning anyone reading the metadata immediately understands which document, which chapter, and which section a chunk belongs to — without needing to query the database.


**Why I Chose Hash-Based Deduplication**

Content-Aware Uniqueness — Instead of relying on filenames or chunk positions, each chunk is fingerprinted by its actual content + metadata combined, meaning the same file uploaded with a different name is still correctly identified as a duplicate.


Performance at Scale — Rather than loading the entire vector database into memory to check for duplicates, we query ChromaDB directly by hash ID, making the deduplication check O(k) where k is only the new chunks being uploaded — not the entire database size. 

Chunk-Level Precision — Deduplication happens at the individual chunk level, not the document level, meaning if a document is updated and only 3 pages change, only those 3 new chunks get re-indexed while unchanged chunks are safely skipped.

This follow happen when user ask question and parent and child behaviour: 

User Query <br>
    ↓
Similarity search on child collection<br>
    ↓
Returns top-k child chunks (each has parent_id in metadata)<br>
    ↓
Extract parent_id from each child<br>
    ↓
Lookup parent store using that parent_id<br>
    ↓
Return parent text as context to LLM<br> 


Section: Retrieval Strategy Decision
Problem I faced:
Irrelevant chunks (2.4, 14.2, E.1) were scoring 
higher than relevant chunks (4.2) because:
- Similarity search only measures vector closeness
- Chunks containing word "chunker" in code scored 
  high regardless of actual topic relevance 

why I reject MMR: 
MMR fixes: Diversity between retrieved chunks
MMR does NOT fix: Why irrelevant chunks score high in the first place

Example:  MMR would still select 2.4 Project Structure
          as Rank 1 (highest score) and diversify 
          FROM it — not remove it. 

Why Reranking wins:

- Takes ALL retrieved chunks
- Scores Query + Chunk TOGETHER
- Understands full context relationship
- 2.4 Project Structure → low score (just file paths)
- 4.2 Header-Based     → high score (actual explanation)



## Bug: Reranker giving low scores to correct chunks

### Query
I asked about 3 topics in one query:
- Similarity Search vs MMR
- Hybrid Search
- Reranking

### What I expected
7.1, 7.2, 7.3 should rank high after reranking.

### What actually happened
F.3 ranked higher than 7.1, 7.2, 7.3 after reranking.

### Root Cause
Similarity search converts my entire query into  single vector.
That vector represents the average meaning of all 3 topics.
So 7.1 only matched one third of that vector.
E.3 matched more because it broadly touched all words.
Reranker also saw the full query vs each chunk — so 7.1 still
only satisfied one-third of the query.

### Fix
Query Decomposition — break the query into three separate 
questions, retrieve for each, then _____ the results.

### problem:

User asks multi-topic query
→ LLM generates sub-queries from imagination
→ Sub-queries don't match document content
→ Wrong chunks retrieved 

### Why it happend: 
LLM had no knowledge of what sections exist in YOUR document
→ Generated generic internet-style queries
→ "parsing algorithms for files" ← not in your doc

### Fix — Retrieval Augmented Query Decomposition:
Step 1: Hybrid search on section names
        (BM25 catches exact keywords + Vector catches semantic meaning)
        → Finds relevant sections from YOUR document

Step 2: Pass those sections to LLM
        → LLM now knows what actually exists
        → Generates grounded sub-queries

Step 3: Sub-queries match real document content
        → Correct chunks retrieved

        