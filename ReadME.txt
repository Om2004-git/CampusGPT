🎓 CampusGPT – AI-Powered College Information Assistant
RAG + LangChain + FAISS + Groq LLM + FastAPI + Streamlit

CampusGPT is an AI-powered assistant designed to help students, faculty, and visitors quickly access verified college information such as:

Faculty list

Admission criteria

Fee structure

Course details

Department info

Campus facilities

Important notices

Academic calendar

And more…

CampusGPT uses RAG (Retrieval Augmented Generation) with FAISS vector search + LLM reasoning to provide accurate answers based on real college documents.

🚀 Features
✔️ RAG-powered Q&A

Uses vector search (FAISS) + Groq LLM to answer questions only from your college documents.

✔️ Secure Document Ingestion

Upload PDFs or ingest an entire directory of documents.

✔️ Smart Retrieval

Automatically retrieves ALL faculty details, department data, or full pages based on the query.

✔️ Clean Streamlit UI

User-friendly web interface for students.

✔️ FastAPI Backend

Production-ready backend with CORS, router structure, environment variables.

✔️ Works Offline after Document Ingestion

Once the FAISS database is created, no internet is needed for vector search.

📁 Project Structure
CampusGPT/
│
├── backend/
│   ├── main.py                     # FastAPI entry point
│   ├── routers/
│   │   ├── gpt.py                  # API endpoints
│   │   └── __init__.py
│   ├── services/
│   │   ├── rag_langchain.py        # RAG Logic (FAISS + LLM)
│   │   └── __init__.py
│   ├── vectorstore/
│   │   └── db_faiss/               # Generated FAISS index
│   ├── data/                        # PDF files for ingestion
│   └── .env                        # environment variables
│
├── streamlit_app.py                # Frontend UI
├── requirements.txt                # Python dependencies
└── README.md

⚙️ Installation & Setup
1️⃣ Create Virtual Environment
python -m venv campus
campus\Scripts\activate   # Windows

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Add your .env inside /backend/
GROQ_API_KEY=your_key_here
GROQ_MODEL_NAME=llama-3.1-8b-instant
DB_FAISS_PATH=backend/vectorstore/db_faiss
UPLOAD_DIR=uploads
ADMIN_TOKEN=hackathon_admin
GROQ_MAX_TOKENS=1500
DEFAULT_K=8

📚 Document Ingestion (Very Important)
Step 1 — Place your PDFs inside:
backend/data/


Example:

backend/data/Faculty.pdf
backend/data/Admission.pdf
backend/data/Syllabus.pdf

Step 2 — Start the backend:
uvicorn backend.main:app --port 8000

Step 3 — Open Swagger docs:

👉 http://127.0.0.1:8000/docs

Step 4 — Ingest all PDFs:

Use endpoint:

POST /gpt/ingest_dir


Body:

admin_token = hackathon_admin


If successful:

{
  "status": "ok",
  "ingested_chunks": 345
}


FAISS index appears at:

backend/vectorstore/db_faiss/index.faiss
backend/vectorstore/db_faiss/index.pkl

🚀 Run CampusGPT Frontend

Open a new terminal:

campus\Scripts\activate
streamlit run streamlit_app.py


Now open:

👉 http://localhost:8501

🧠 How CampusGPT Works

PDFs → chunked → embedded using HuggingFace (MiniLM-L6-v2)

Stored in a FAISS vector database

Query → converted to embedding

Top k matching chunks retrieved
(Smart mode for faculty queries → retrieves ALL relevant chunks)

Prompt created using retrieved context

Groq LLM (llama-3.1-8b-instant) generates reliable answer

💡 Example Queries

“Give all faculty names of Computer Engineering Department”

“What is the eligibility for B.Tech admission?”

“Give me complete fee structure for all courses”

“List all labs in Mechanical Engineering”

“What documents are required for admission?”