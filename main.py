from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services import process_urls, ask_question

app = FastAPI(title="ElopyBot News Research API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processing_status = {"ready": True}

class UrlsInput(BaseModel):
    urls: list[str]

class Question(BaseModel):
    query: str

@app.get("/")
async def home():
    return {"message": "ElopyBot API is running 🚀"}

@app.post("/process/")
async def process_urls_endpoint(data: UrlsInput):
    processing_status["ready"] = False
    chunks = process_urls(data.urls)
    processing_status["ready"] = True
    return {"message": f"Processed {chunks} chunks from {len(data.urls)} URLs."}

@app.post("/ask/")
async def ask_question_endpoint(question: Question):
    if not processing_status["ready"]:
        return {"answer": "Still processing. Please wait..."}
    result = ask_question(question.query)
    return result
