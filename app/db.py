from chromadb import PersistentClient

chroma_client = PersistentClient(path="../chroma")

def setup_chroma():
    chroma_client.get_or_create_collection(name="rag-store")
    print("✅ Chroma DB initialized and ready")