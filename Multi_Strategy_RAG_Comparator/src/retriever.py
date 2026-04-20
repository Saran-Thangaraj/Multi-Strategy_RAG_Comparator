import re
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def is_multi_topic(query: str) -> bool:
    has_bullets = re.search(r'[-•]\s', query)
    has_commas = re.search(r'[,]\s', query)
    has_vs = re.search(r'\b(vs\.?)', query)
    has_compare = re.search(r'\b(compare|comparison|difference between|contrast)\b', query, re.IGNORECASE)
    has_ampersand = re.search(r'\s&\s', query)
    return bool(has_bullets or has_commas or has_vs or has_compare or has_ampersand)


def extract_sections(chunks: list) -> list:
    seen = set()
    sections = []
    for chunk in chunks:
        # parent_child → parent_id
        # header       → header_id
        # fixed        → Header 1
        label = (
            chunk.metadata.get("parent_id") or
            chunk.metadata.get("header_id") or
            chunk.metadata.get("Header 1", "") or 
            chunk.metadata.get("page_chapter", "")
        )
        if label and label not in seen:
            seen.add(label)
            sections.append(label)
    return sections


def get_relevant_sections(query: str, chunks: list, embedding) -> list:
    sections = extract_sections(chunks)
    section_docs = [Document(page_content=section) for section in sections]

    bm25 = BM25Retriever.from_documents(section_docs)
    bm25.k = 5

    mini_vectorstore = Chroma.from_documents(
        documents=section_docs,
        embedding=embedding
    )
    vector = mini_vectorstore.as_retriever(search_kwargs={"k": 5})

    hybrid = EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[0.4, 0.6]
    )

    results = hybrid.invoke(query)
    return [doc.page_content for doc in results]


decompose_prompt = PromptTemplate(
    input_variables=["question", "sections"],
    template="""
You are an expert search query planner for a RAG system.

Available sections in the document:
{sections}

Generate 4-5 natural sub-queries based ONLY on the above sections.
Rules:
- Use section names as guidance
- Do NOT invent topics not in sections
- Return ONLY comma-separated list

User question:
{question}

Sub-questions:"""
)

def decompose_and_retrieve(query: str, embedding, chunks: list, llm, retrieval_embedding) -> list:
    parser = CommaSeparatedListOutputParser()

    if is_multi_topic(query):
        qe = get_relevant_sections(query, chunks, embedding)

        print("qe",qe)

        allowed_chapters = set()
        for section in qe:
            parts = section.split("-")
            if len(parts) > 1:
                allowed_chapters.add(parts[1])

        chain = decompose_prompt | llm | parser
        sub_questions = chain.invoke({
            "question": query,
            "sections": "\n".join(qe)
        })
    else:
        sub_questions = [query]
        allowed_chapters = None

    print(sub_questions)

    sub_results = []
    for sub_q in sub_questions:

        # Build search kwargs dynamically
        search_kwargs = {"k": 3}
        if allowed_chapters:
            search_kwargs["filter"] = {"Header 1": {"$in": list(allowed_chapters)}}

        # Use as_retriever — works for all 3 strategies
        retriever = retrieval_embedding.as_retriever(search_kwargs=search_kwargs)
        answer = retriever.invoke(sub_q)
        sub_results.append((sub_q, answer))

    # Deduplicate
    seen = set()
    unique_context = []
    for sub_q, docs in sub_results:
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_context.append(doc)

    return unique_context