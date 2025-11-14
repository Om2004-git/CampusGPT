# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import gpt

app = FastAPI(
    title="CampusGPT AI Backend",
    description="RAG-powered AI for College Automation",
    version="1.0.0",
)

# --------------------------
# Enable CORS (important for Android / Web)
# --------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Include Routers
# --------------------------
app.include_router(gpt.router, prefix="/gpt")


@app.get("/")
def root():
    return {"status": "ok", "message": "CampusGPT Backend Running Successfully!"}
