"""
app.py
------
Streamlit chat UI for asking questions over your PDFs using RAG:
Chroma (retriever, local embeddings) + Google Gemini for answers.

Run:
    streamlit run app.py

Prerequisite: run `python ingest.py` first to build the Chroma DB.
"""

import os

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    LLM_MODEL,
    GOOGLE_API_KEY,
    TOP_K,
    check_api_key_ready,
)

st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="wide")
st.title("📄 Chat with your PDFs (RAG + Gemini)")


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner="Loading vector store...")
def load_vectorstore(_embeddings):
    if not os.path.isdir(CHROMA_DIR):
        return None
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings,
        persist_directory=CHROMA_DIR,
    )


@st.cache_resource(show_spinner="Connecting to Gemini...")
def load_llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )


def format_docs(docs):
    parts = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        parts.append(f"[Source: {source}, page {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below,
which was retrieved from the user's PDF documents.

- If the answer isn't in the context, say you don't know based on the provided documents.
- Be concise and accurate.
- Cite the source filename and page number when relevant.

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Startup checks + load resources
# ---------------------------------------------------------------------------
check_api_key_ready()

embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)
if vectorstore is None or vectorstore._collection.count() == 0:
    st.error(
        f"No Chroma DB found (or it's empty) at './{CHROMA_DIR}'.\n\n"
        "Run `python ingest.py` first to build the database from your PDFs."
    )
    st.stop()

llm = load_llm()
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ask something about your PDFs...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = retriever.invoke(user_question)
            answer = rag_chain.invoke(user_question)
            st.markdown(answer)

            with st.expander("Sources"):
                for d in docs:
                    source = d.metadata.get("source", "unknown")
                    page = d.metadata.get("page", "?")
                    st.markdown(f"**{source}** — page {page}")
                    st.caption(d.page_content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("Settings")
    st.write(f"**LLM model:** `{LLM_MODEL}`")
    st.write(f"**Embedding model:** `{EMBED_MODEL}` (local)")
    st.write(f"**Chroma DB:** `{CHROMA_DIR}` (collection: `{COLLECTION_NAME}`)")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
