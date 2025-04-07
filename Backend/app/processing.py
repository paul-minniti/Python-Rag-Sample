from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import openai
from collections import defaultdict

chat_history = defaultdict(list)


def get_history(chat_name: str) -> list[dict]:
    return chat_history[chat_name]


def load_documents(data_dir: str, file_name: str):
    loader = TextLoader(data_dir + "/" + file_name)
    # loader = DirectoryLoader(data_dir, glob=file_name)
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def build_prompt(query: str, context_docs: list):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = (
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return prompt


def ask_gpt(chat_name: str, prompt: str, model: str = "gpt-4o"):
    history = get_history(chat_name)
    history.append({"role": "user", "content": prompt})
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."}
        ] + history
    )
    answer = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": answer})
    return answer