import os
from dotenv import load_dotenv
from chromadb.config import Settings
from chromadb import Client
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

dbpath = './chromadb'
client = Client(Settings())


def get_vector_store(split_docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    collection = client.create_collection(name="documents")

    for idx, chunk in enumerate(split_docs):
        collection.add(
            documents=[chunk.page_content], 
            metadatas=[{'id': idx}], 
            embeddings=embeddings.embed_documents([chunk.page_content]), 
            ids=[str(idx)] 
        )
    vector_store = Chroma(
        collection_name="documents",
        embedding_function=embeddings
    )
    return vector_store
