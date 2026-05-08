# nodes/decision_node.py
# Decision and answer agents
# grade_node              → are the chunks good enough? pass / fail / exhausted
# strategy_selector_node  → LLM reasons which strategy to try next (called on fail)
# answer_node             → generate final answer from good chunks

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

STRATEGIES = ["header", "child", "fixed"]

MIN_QUALITY_THRESHOLD = 0.35


# ── Grade Node ────────────────────────────────────────────────
# One job: decide if retrieved chunks are good enough to answer
# Does NOT pick the next strategy — that is strategy_selector_node's job
#
# Writes: state["grade"]           = "pass" / "fail" / "exhausted"
#         state["avg_score"]       = float (read by strategy_selector_node)
#         state["reranked_chunks"] = filtered good chunks (on pass)
#         state["retry_count"]     = incremented (on fail)

def grade_node(state: dict) -> dict:
    chunks = state["reranked_chunks"]

    # mark current strategy as tried before evaluating
    tried = state["tried_strategies"]
    if state["current_strategy"] not in tried:
        tried.append(state["current_strategy"])

    if state["retry_count"] >= state["max_retry"]:
        state["grade"]  = "exhausted"
        state["answer"] = (
            "I could not find relevant information for your query. "
            "Please try rephrasing your question."
        )
        return state

    scores    = [doc.metadata.get("relevance_score", 0) for doc in chunks]
    print("Relevance scores:", scores)

    avg_score = sum(scores) / len(scores) if scores else 0
    print("Average score:", avg_score)
    state["avg_score"] = avg_score  # stored for strategy_selector_node

    if avg_score < MIN_QUALITY_THRESHOLD:
        remaining = [s for s in STRATEGIES if s not in tried]
        if not remaining:
            state["grade"]  = "exhausted"
            state["answer"] = (
                "I could not find relevant information for your query. "
                "Please try rephrasing your question."
            )
            print(f"Grade: EXHAUSTED — all strategies tried: {tried}")
            return state

        state["retry_count"] += 1
        state["grade"]        = "fail"
        print(f"Grade: FAIL (avg {avg_score:.3f} < {MIN_QUALITY_THRESHOLD}) — tried so far: {tried}")
        return state

    good_chunks = [doc for doc in chunks if doc.metadata.get("relevance_score", 0) >= avg_score]
    if good_chunks:
        state["grade"]           = "pass"
        state["reranked_chunks"] = good_chunks
    return state


# ── Strategy Selector Node ────────────────────────────────────
# One job: LLM reasons why the last strategy failed and picks the best next one
# Only called when grade = "fail"
#
# Writes: state["current_strategy"] = next strategy chosen by LLM

strategy_prompt = PromptTemplate(
    input_variables=["query", "failed_strategy", "avg_score", "sample_chunks", "remaining_strategies"],
    template="""You are a RAG retrieval strategy selector.

A retrieval attempt just failed to find sufficiently relevant chunks.

Query: {query}
Failed strategy: {failed_strategy}
Average relevance score achieved: {avg_score} (minimum required: 0.35)

Sample of chunks retrieved (showing why it failed):
{sample_chunks}

Remaining strategies to choose from: {remaining_strategies}

Strategy guide:
- header : Chunks split by document headers/sections. Best for topic-specific or section-based questions.
- fixed  : Uniform fixed-size chunks. Best for code snippets, precise technical details, uniform content.
- child  : Small child chunks backed by parent context. Best for broad conceptual questions needing surrounding text.

Reason: Why did "{failed_strategy}" fail for this query? Which remaining strategy is most likely to succeed?
Reply with ONE word only from: {remaining_strategies}

Next strategy:"""
)


def strategy_selector_node(state: dict, llm) -> dict:
    failed_strategy = state["current_strategy"]
    avg_score       = state.get("avg_score", 0)
    tried           = state["tried_strategies"]
    remaining       = [s for s in STRATEGIES if s not in tried]

    # only one option left — skip LLM call
    if len(remaining) == 1:
        state["current_strategy"] = remaining[0]
        print(f"Strategy selector: one option left → {remaining[0]}")
        return state

    chunks = state.get("reranked_chunks", [])[:2]
    sample_chunks = "\n---\n".join(
        f"[score: {doc.metadata.get('relevance_score', 'N/A')}]\n{doc.page_content[:300]}"
        for doc in chunks
    ) or "No chunks retrieved."

    chain    = strategy_prompt | llm | StrOutputParser()
    response = chain.invoke({
        "query":                state["query"],
        "failed_strategy":      failed_strategy,
        "avg_score":            f"{avg_score:.3f}",
        "sample_chunks":        sample_chunks,
        "remaining_strategies": ", ".join(remaining),
    })

    picked = response.strip().lower().split()[0]
    if picked not in remaining:
        picked = remaining[0]  # fallback if LLM output is unexpected

    state["current_strategy"] = picked
    print(f"Strategy selector: '{failed_strategy}' failed → LLM picked '{picked}' from {remaining}")
    return state


# ── Answer Node ───────────────────────────────────────────────
# One job: generate final answer from good chunks
# Only called when grade = "pass"
# Writes: state["answer"] = final answer string

answer_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""Answer the question using ONLY the context below. Do NOT use any outside knowledge.

Rules:
- Include complete code examples exactly as they appear in context
- Do not summarize or shorten code blocks
- Structure your answer clearly
- If the context does not contain enough information to answer the question, respond with exactly:
  "I could not find a relevant answer to your question in the provided document."
- Always wrap all code examples in triple backticks ```python ... ```

Context:
{context}

Question: {query}
Answer:"""
)

def answer_node(state: dict, llm) -> dict:
    query   = state["query"]
    chunks  = state["reranked_chunks"]
    context = "\n\n".join([doc.page_content for doc in chunks])

    chain           = answer_prompt | llm | StrOutputParser()
    state["answer"] = chain.invoke({"query": query, "context": context})
    return state
