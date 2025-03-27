from chromadb import PersistentClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

chroma_client = PersistentClient(path="../chroma")

def setup_chroma():
    chroma_client.get_or_create_collection(name="rag-store")
    print("✅ Chroma DB initialized and ready")


def save_to_chroma(chunks: list[Document]):
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory="chroma"
    )
    db.persist()
    print(f"✅ {len(chunks)} chunks saved to Chroma.")