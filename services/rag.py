import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_ollama.chat_models import ChatOllama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_ollama.embeddings import OllamaEmbeddings

load_dotenv()

dbpath = './chromadb'
embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", "mxbai-embed-large"))

template = """You are an assistant for an education institute called EduProvider.  
Use the provided CONTEXT to answer the user’s question clearly and concisely. 

Rules:
- Give one clear, concise answer only.  
- If the CONTEXT contains the answer, provide it directly.  
- Do NOT show reasoning, analysis, or explanations of what the context does/does not include.  
- If the CONTEXT does not contain the answer or the question is unrelated to courses, admissions, or student support, 
  politely redirect the user to call EduProvider at +1-333-111-2345.  
- Never prefix with 'Assistant:' or 'Answer:'. Just respond directly.  

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

def save_to_vector_store(docs):
    """Save documents to vector database

    Args:
        docs: Loaded documents.

    Returns:
        vector_store: Vector store object.
    """
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=dbpath,
    )
    print(f"Saved {len(docs)} documents to vector store at {dbpath}")
    return vector_store


def build_rag(vector_store):
    """Build the RAG

    Args:
        vector_store: Vector store object.

    Returns:
        qa: RAG object.
    """
    # Use MMR (Maximal Marginal Relevance) to reduce duplicate chunks
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": int(os.getenv("TOP_K", 4)),
            "fetch_k": int(os.getenv("FETCH_K", 20)),
            "lambda_mult": float(os.getenv("LAMBDA_MULT", 0.5))
        }
    )
    llm = ChatOllama(model=os.getenv("MODEL_NAME", "deepseek-r1:1.5b"), temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return qa
