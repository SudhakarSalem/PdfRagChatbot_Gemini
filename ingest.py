"""
ingest.py
---------
Reads every PDF from PDF_DIR (default D:\\pdf), splits into chunks,
embeds them with a local sentence-transformers model (no API calls),
and stores them in a persistent Chroma vector database on disk.

Run this once, and again any time you add/change PDFs:

    python ingest.py
"""

import sys
import shutil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    PDF_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_pdfs(pdf_dir: str):
    pdf_dir_path = Path(pdf_dir)
    if not pdf_dir_path.exists():
        print(f"ERROR: PDF folder not found: {pdf_dir}")
        sys.exit(1)

    pdf_files = sorted(pdf_dir_path.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: No PDF files found in: {pdf_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF file(s) in {pdf_dir}")

    all_docs = []
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = pdf_path.name
            all_docs.extend(docs)
        except Exception as e:
            print(f"  WARNING: failed to load {pdf_path.name}: {e}")

    return all_docs


def main():
    print("=" * 60)
    print("PDF RAG Ingestion (Gemini + local embeddings + Chroma)")
    print("=" * 60)
    print(f"PDF folder      : {PDF_DIR}")
    print(f"Embedding model : {EMBED_MODEL}  (runs locally, no API key needed)")
    print(f"Chroma DB dir   : ./{CHROMA_DIR}")
    print("=" * 60)

    docs = load_pdfs(PDF_DIR)
    print(f"Loaded {len(docs)} total pages")

    if not docs:
        print("ERROR: No documents were loaded (PDFs may be empty/corrupt).")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    print("Loading local embedding model (first run downloads it, ~90MB)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db_path = Path(CHROMA_DIR)
    if db_path.exists():
        print(f"Removing existing Chroma DB at ./{CHROMA_DIR} for a clean rebuild...")
        shutil.rmtree(db_path)

    print("Embedding chunks and writing to Chroma... (this can take a while)")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print()
    print(f"Saved Chroma DB to: ./{CHROMA_DIR}  (collection: {COLLECTION_NAME})")
    print("Done. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
