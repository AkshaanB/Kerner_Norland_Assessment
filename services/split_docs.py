import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split the documents into chunks

    Args:
        docs: Loaded documents.
        chunk_size (int, optional): Size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 200.

    Returns:
        _type_: Split documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", chunk_size)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", chunk_overlap)),
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Splited documents into {len(split_docs)} chunks.")
    return split_docs
