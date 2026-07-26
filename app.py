"""
MedRAG - Medical Research Assistant (Streamlit UI)

Upload PDFs, build a vector store, and chat with an LLM that answers
ONLY from the retrieved context. Chat history persists in the session.

Run with:
    streamlit run app.py
"""

import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma  # pip install -U langchain-chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # must match
                                                                   # everywhere
PERSIST_DIR = "Medical-db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

SYSTEM_PROMPT = """
You are MedRAG, an AI-powered Medical Research Assistant.

Your role is to answer users' questions using ONLY the retrieved medical research documents provided to you.

Rules:
1. Use ONLY the retrieved context to answer.
2. Never use your own knowledge.
3. Never fabricate or guess information.
4. If the answer is not present in the retrieved documents, say:
   "I could not find sufficient evidence in the retrieved medical documents to answer this question."
5. If multiple papers contain different findings, mention all viewpoints objectively.
6. Summarize information from multiple papers when appropriate.
7. Keep the answer scientifically accurate, concise, and easy to understand.
8. Never provide personal medical advice or make clinical recommendations.
9. Clearly mention uncertainty if the evidence is limited.
10. At the end of every response, include a Sources section.

Response Format:

## Summary
A concise answer.

## Detailed Explanation
Explain the findings using the retrieved evidence.

## Sources
List the paper titles (and page numbers if available).
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
Retrieved Context:

{context}

----------------------------------------

User Question:

{question}
""",
        ),
    ]
)

st.set_page_config(page_title="MedRAG - Medical Research Assistant", page_icon="🩺", layout="wide")


# ---------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-2506")


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


def build_vectorstore(uploaded_files, persist_dir: str):
    """Save uploaded PDFs to a temp dir, load, chunk, embed, and persist to Chroma."""
    embedding_model = get_embedding_model()

    # Wipe any old DB so re-uploads don't mix with a stale collection
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    all_docs = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # tag each page with its source filename for citation purposes
            for d in docs:
                d.metadata["source_file"] = uploaded_file.name
            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(all_docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    return vectorstore, len(all_docs), len(chunks)


def format_docs_for_context(docs):
    parts = []
    for d in docs:
        source = d.metadata.get("source_file", "Unknown")
        page = d.metadata.get("page", "N/A")
        parts.append(f"[Source: {source} | Page: {page}]\n{d.page_content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------
# Sidebar - upload & build
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process PDFs", type="primary", use_container_width=True, disabled=not uploaded_files):
        with st.spinner("Reading, chunking, and embedding documents... this can take a while"):
            vectorstore, n_pages, n_chunks = build_vectorstore(uploaded_files, PERSIST_DIR)
            st.session_state.vectorstore = vectorstore
            st.session_state.processed_files = [f.name for f in uploaded_files]
        st.success(f"Indexed {n_pages} pages into {n_chunks} chunks.")

    if st.session_state.processed_files:
        st.markdown("**Indexed files:**")
        for name in st.session_state.processed_files:
            st.markdown(f"- {name}")

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "Answers are generated ONLY from the uploaded PDFs. "
        "This tool does not provide medical advice."
    )


# ---------------------------------------------------------------------
# Main chat UI
# ---------------------------------------------------------------------
st.title("🩺 MedRAG — Medical Research Assistant")
st.caption("Upload medical research PDFs on the left, then ask questions below.")

# render existing history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about the uploaded documents...")

if query:
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process at least one PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages and generating answer..."):
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5},
                )
                docs = retriever.invoke(query)
                context = format_docs_for_context(docs)

                final_prompt = PROMPT.invoke({"context": context, "question": query})
                llm = get_llm()
                response = llm.invoke(final_prompt)

            st.markdown(response.content)

        st.session_state.messages.append({"role": "assistant", "content": response.content})