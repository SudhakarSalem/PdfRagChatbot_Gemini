"""
config.py
---------
Shared configuration for the PDF RAG chatbot. Uses:
  - Google Gemini (free API tier) for answer generation
  - A local sentence-transformers model for embeddings (free, runs on
    your machine, no API calls/cost)
  - Chroma as the vector database
"""

import os
import sys

# ---------------------------------------------------------------------------
# Configuration (override any of these with environment variables)
# ---------------------------------------------------------------------------
PDF_DIR = os.environ.get("PDF_DIR", r"D:\pdf")

CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "pdf_docs")

# Local embedding model (downloaded once from Hugging Face on first run,
# then cached — no API key needed, no per-call cost).
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Gemini model used for answering questions. "gemini-flash-latest" is a
# Google-maintained alias that always points at their current recommended
# fast/free-tier-friendly Flash model, so this keeps working even as Google
# renames or upgrades the underlying model.
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-flash-latest")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4


def check_api_key_ready():
    """Fail fast with a clear message if GOOGLE_API_KEY isn't set."""
    if not GOOGLE_API_KEY:
        print("=" * 60)
        print("ERROR: GOOGLE_API_KEY environment variable is not set.")
        print()
        print("Fix — set it before running, e.g. (PowerShell):")
        print('    $env:GOOGLE_API_KEY="AIza..."')
        print()
        print("Or create a .env file in this folder containing:")
        print("    GOOGLE_API_KEY=AIza...")
        print()
        print("Get a free key at: https://aistudio.google.com/apikey")
        print("=" * 60)
        sys.exit(1)
