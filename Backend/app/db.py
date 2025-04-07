from chromadb import PersistentClient
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma

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


def query_chroma(query: str, k: int = 25):
    db = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory="chroma",
    )
    results = db.similarity_search(query, k=k)
    
    if not results:
        raise ValueError("No relevant documents found in the vector store.")

    return results