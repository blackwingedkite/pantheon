import chromadb
import uuid
import os
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
load_dotenv()
EMBEDDING_MODEL = "text-embedding-3-small"
embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name=EMBEDDING_MODEL)

def get_or_create_collection(collection_name):
    client = chromadb.PersistentClient(path=collection_name)  # 或 HttpClient()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    return collection

def add_chunk_to_db(chunks, collection):
    # 將 chunks 添加到 Chroma 集合中
    print(len(chunks))
    for chunk in chunks:
        print(chunk["chunk"][0:20])
        unique_id = str(uuid.uuid4())  # 生成唯一的ID
        collection.add(
            documents=[chunk["chunk"]],
            metadatas=[{"start_page": chunk["start_page"], "end_page": chunk["end_page"]}],
            ids=[unique_id]
        )