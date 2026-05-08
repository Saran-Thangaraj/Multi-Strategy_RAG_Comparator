from main import build_pipeline, run_query
from langchain_groq import ChatGroq
import streamlit as st
import tempfile, os

st.title("Multi-Strategy RAG Comparator")

with st.sidebar:
    groq_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    cohere_key = st.sidebar.text_input("Enter Cohere API Key", type="password")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file and st.button("Build Pipeline"):
        # Save with consistent name — not random temp name
        UPLOAD_PATH = "uploaded_doc.pdf"
        with open(UPLOAD_PATH, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Building pipeline..."):
            st.session_state.pipeline = build_pipeline(pdf_path=UPLOAD_PATH)

        st.success(f"Pipeline built from: {uploaded_file.name}")

    if "pipeline" in st.session_state:
        st.success("Pipeline is ready!")

if groq_key and "llm" not in st.session_state:
    os.environ['GROQ_API_KEY'] = groq_key
    os.environ['CO_API_KEY'] = cohere_key
    st.session_state.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=groq_key)

query = st.text_input("Enter your question:")

if st.button("Search"):
    if not query:
        st.warning("Please enter a question.")
    elif "pipeline" not in st.session_state:
        st.warning("Please upload a PDF and build the pipeline first.")
    else:
        with st.spinner("Searching..."):
            results = run_query(query, *st.session_state.pipeline, st.session_state.llm)

        st.subheader("Results")
        for strategy_name, docs in results.items():
            st.markdown(f"### Strategy: {strategy_name}")

            if not docs:
                st.warning("No Relevent chunks found for this strategy")
            else:
                for i, doc in enumerate(docs, 1):
                    section = (
                        doc.metadata.get("Header 2")
                        or doc.metadata.get("Header 1")
                        or doc.metadata.get('section_id')
                        or doc.metadata.get('chapter_id')
                        or doc.metadata.get("page_chapter")
                        or "Unknown"
                    )
                    score = doc.metadata.get("relevance_score", "N/A")
                    st.markdown(f"**Result {i}** — Section: `{section}` | Score: `{score}`")
                    with st.expander("View chunk content"):
                        st.text(doc.page_content)
            st.divider()
