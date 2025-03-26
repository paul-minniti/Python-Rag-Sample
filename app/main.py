from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import shutil
import uvicorn

app = FastAPI()
DATA_DIR = "data"
CHROMA_DIR = "chroma"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

@app.get("/")
async def read_root():
    key = os.environ['OPENAI_API_KEY']
    return {"api-key": key}

@app.post("/upload/")
async def upload_file(file: UploadFile = File()):
    file_location = os.path.join(DATA_DIR, file.filename)
    
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")
    finally:
        file.file.close()

    return {"info": f"File '{file.filename}' saved at '{file_location}'"}

def run_server():
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    run_server()
