# backend/services/rag_langchain.py

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# --------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------------
load_dotenv()   # IMPORTANT: loads .env

# DEBUG (optional)
print("🔑 Loaded GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

# --------------------------------------------------------
# FIXED: SAFE PATH HANDLING
# --------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]   # → CampusGPT/
RELATIVE_DB_PATH = os.getenv("DB_FAISS_PATH", "backend/vectorstore/db_faiss")
DB_FAISS_PATH = (ROOT_DIR / RELATIVE_DB_PATH).resolve()

os.makedirs(DB_FAISS_PATH, exist_ok=True)

# --------------------------------------------------------
# IMPORTS
# --------------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")              # MUST NOT be None
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
DEFAULT_K = int(os.getenv("DEFAULT_K", "8"))

# VALIDATE API KEY
if not GROQ_API_KEY:
    raise ValueError("❌ ERROR: GROQ_API_KEY is missing. Check your .env file.")


# --------------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------------
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# --------------------------------------------------------
# INGEST DIRECTORY
# --------------------------------------------------------
def ingest_directory(data_path: str = "data/", persist_path: Path = DB_FAISS_PATH,
                     chunk_size: int = 500, chunk_overlap: int = 50):

    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    if not docs:
        return {"status": "no_docs", "count": 0}

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    embeddings = get_embedding_model()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(persist_path))

    return {"status": "ok", "ingested_chunks": len(chunks)}


# --------------------------------------------------------
# INGEST SINGLE FILE
# --------------------------------------------------------
def ingest_file(file_path: str, persist_path: Path = DB_FAISS_PATH,
                chunk_size: int = 600, chunk_overlap: int = 70):

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    if not pages:
        return {"status": "no_content", "ingested": 0}

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(pages)

    embeddings = get_embedding_model()

    if persist_path.exists() and any(persist_path.iterdir()):
        db = FAISS.load_local(str(persist_path), embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(str(persist_path))
    return {"status": "ok", "ingested_chunks": len(chunks)}


# --------------------------------------------------------
# PROMPT
# --------------------------------------------------------
RAG_PROMPT = """
You are **CampusGPT**, the official AI assistant for the college.

Use ONLY the following context extracted from college documents.
If the answer is not found in the context, say:
"I could not find this information in the college documents."

Be detailed. Return every important detail.
Do NOT skip names. Do NOT summarize unless asked.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""

prompt = PromptTemplate(template=RAG_PROMPT, input_variables=["question", "context"])


# --------------------------------------------------------
# QUERY RAG (SMART + HARDCODED LOGIC)
# --------------------------------------------------------
def query_rag(query: str, k: int = DEFAULT_K):

    lower_q = query.lower()

    # =====================================================
    # 1) HARDCODED DIRECTOR RULE
    # =====================================================
    DIRECTOR_KEYWORDS = [
        "director",
        "who is the director",
        "director name",
        "principal",
        "head of institute",
        "hod director",
        "campus director"
    ]

    if any(key in lower_q for key in DIRECTOR_KEYWORDS):
        return {
            "answer": "The Director of TIT Bhiwani is **Prof. Dr. B.K Behera**, serving as the *Director of TIT Bhiwani*.",
            "sources": [
                {
                    "filename": "Hardcoded Rule",
                    "snippet": "Prof. Dr. B.K Behera is the Director of TIT Bhiwani."
                }
            ]
        }

    # =====================================================
    # 2) LOAD VECTOR DATABASE
    # =====================================================
    embeddings = get_embedding_model()
    db = FAISS.load_local(str(DB_FAISS_PATH), embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k": k})

    # =====================================================
    # 3) SMART FACULTY RETRIEVAL
    # =====================================================
    FACULTY_KEYWORDS = [
        "faculty", "faculties", "teachers", "staff",
        "professor", "professors", "hod", "department faculty",
        "teaching staff", "assistant professor", "associate professor"
    ]

    if any(word in lower_q for word in FACULTY_KEYWORDS):

        # Fetch ALL chunks
        all_docs = db.similarity_search("", k=9999)

        # Prefer Faculty.pdf
        faculty_docs = [d for d in all_docs if "Faculty" in d.metadata.get("source", "")]

        docs = faculty_docs if faculty_docs else all_docs

    else:
        docs = retriever.invoke(query)

    # =====================================================
    # 4) FORMAT CONTEXT + RUN LLM
    # =====================================================
    context_text = "\n\n".join([d.page_content for d in docs])

    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=GROQ_API_KEY,      # <— MUST BE PASSED
        temperature=0.5,
        max_tokens=1500
    )

    final_prompt = prompt.format(question=query, context=context_text)
    parser = StrOutputParser()
    answer = (llm | parser).invoke(final_prompt)

    # =====================================================
    # 5) RETURN SOURCES
    # =====================================================
    sources = []
    for d in docs:
        filename = d.metadata.get("source", d.metadata.get("path", "Unknown"))
        snippet = d.page_content[:800]
        sources.append({
            "filename": filename,
            "snippet": snippet
        })

    return {"answer": answer, "sources": sources}
