# Test each service individually

from services.load_docs import load_documents
from services.split_docs import split_documents
from services.rag import save_to_vector_store, build_rag


def load_text_document(folder_path):
    documents = load_documents(folder_path)
    assert len(documents) > 0
    return documents


def split_text_documents(documents):
    split_docs = split_documents(documents)
    assert len(split_docs) > 0
    return split_docs


def test_create_vector_store(split_docs):
    vector_store = save_to_vector_store(split_docs)
    assert vector_store is not None
    return vector_store


def build_rag_test(vector_store):
    qa = build_rag(vector_store)
    assert qa is not None
    return qa


if __name__ == "__main__":
    documents = load_documents("data/")
    print(f"Total loaded documents: {len(documents)}")
    split_docs = split_documents(documents)
    print(f"Total split documents: {len(split_docs)}")
    vector_store = save_to_vector_store(split_docs)
    print("Vector store created successfully.")
    qa = build_rag(vector_store)
    print("RAG model built successfully.")