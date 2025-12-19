import glob
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader

def load_documents(folder="./data"):
    """Load the documents

    Args:
        folder (str, optional): Folder which the data is stored in. Defaults to "./data".

    Returns:
        docs: The loaded documents.
    """
    docs = []
    for ext in ("*.txt", "*.md"):
        for file in glob.glob(str(Path(folder)/"**"/ext), recursive=True):
            docs.extend(TextLoader(file).load())
    for file in glob.glob(str(Path(folder)/"**/*.pdf"), recursive=True):
        docs.extend(PyPDFLoader(file).load())
    print(f"Loaded {len(docs)} documents from {folder}")
    return docs
