from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import shutil
import uvicorn
from dotenv import load_dotenv
import openai 
from pydantic import BaseModel
from contextlib import asynccontextmanager
from app.db import setup_chroma, save_to_chroma, query_chroma
from app.processing import load_documents, split_text, build_prompt, ask_gpt

DATA_DIR = "data"
CHROMA_DIR = "chroma"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_chroma()
    yield
    print("🛑 FastAPI is shutting down")

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    key = os.environ['OPENAI_API_KEY']
    return {"api-key": key}


class ChatRequest(BaseModel):
    query: str

@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        print(f"💬 Query received: {request.query}")
        context_docs = query_chroma(request.query)
        print(f"📚 Retrieved {len(context_docs)} context docs from Chroma.")
        prompt = build_prompt(request.query, context_docs)
        answer = ask_gpt(prompt)
        print(f"prompt generated: {prompt}")
        citations = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "snippet": doc.page_content[:200] + "…" if len(doc.page_content) > 200 else doc.page_content
            }
            for doc in context_docs
        ]
        return {"query": request.query, "response": answer, "citations": citations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(DATA_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")
    finally:
        file.file.close()

    try:
        print(f"Uploading file: {file.filename}")
        documents = load_documents(DATA_DIR, file.filename)
        print(f"Loaded {len(documents)} documents.")
        chunks = split_text(documents)
        save_to_chroma(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    return {"filename": file.filename, "status": "✅ File uploaded and indexed"}

def run_server():
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    run_server()
