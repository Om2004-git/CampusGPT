from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from backend.services.rag_langchain import ingest_file, ingest_directory, query_rag

router = APIRouter()


# -----------------------------
# JSON Model for /ask
# -----------------------------
class AskRequest(BaseModel):
    query: str


# -----------------------------
# Query RAG (POST /gpt/ask)
# -----------------------------
@router.post("/ask")
def ask_question(req: AskRequest):
    """
    Expected JSON:
    {
        "query": "your question"
    }
    """
    return query_rag(req.query)


# -----------------------------
# Ingest Single PDF
# -----------------------------
@router.post("/ingest")
async def ingest_pdf(
    file: UploadFile = File(...),
    admin_token: str = Form(...)
):
    if admin_token != "hackathon_admin":
        return {"status": "error", "msg": "Invalid admin token"}

    # Save PDF
    file_loc = f"data/{file.filename}"
    with open(file_loc, "wb") as f:
        f.write(await file.read())

    # Ingest PDF
    return ingest_file(file_loc)


# -----------------------------
# Ingest ALL PDFs in data/
# -----------------------------
@router.post("/ingest_dir")
def ingest_dir(admin_token: str = Form(...)):
    if admin_token != "hackathon_admin":
        return {"status": "error", "msg": "Invalid admin token"}

    return ingest_directory("data/")
