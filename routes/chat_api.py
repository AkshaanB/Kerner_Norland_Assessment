import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from services.load_docs import load_documents
from services.rag import save_to_vector_store, build_rag
from services.clean import strip_think

load_dotenv()

app = FastAPI(title="EduProvider Institute Chatbot API")

class Query(BaseModel):
    question: str

class Source(BaseModel):
    doc: str
    chunk_id: int
    score: float
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]

docs = load_documents()
vs = save_to_vector_store(docs)
qa = build_rag(vs)

@app.get("/")
def root():
    return {"message": "Welcome to the EduProvider Institute Chatbot API."}

@app.post("/chat", response_model=ChatResponse)
def chat(query: Query):
    
    ans = qa.invoke(query.question)
    cleaned = strip_think(ans["result"])

    # Use MMR search to reduce duplicate chunks
    top_k = int(os.getenv("TOP_K", 4))
    fetch_k = int(os.getenv("FETCH_K", 20))
    lambda_mult = float(os.getenv("LAMBDA_MULT", 0.5))

    mmr_docs = vs.max_marginal_relevance_search(
        query.question,
        k=top_k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )

    all_docs_with_scores = vs.similarity_search_with_score(query.question, k=fetch_k)

    docs_with_scores = []
    for mmr_doc in mmr_docs:
        # Find matching document in scored results
        for scored_doc, score in all_docs_with_scores:
            if scored_doc.page_content == mmr_doc.page_content:
                docs_with_scores.append((mmr_doc, score))
                break
        else:
            # If no exact match found, assign a default score
            docs_with_scores.append((mmr_doc, 0.0))

    sources = []
    for idx, (doc, score) in enumerate(docs_with_scores):
        source = Source(
            doc=doc.metadata.get("source", "unknown"),
            chunk_id=idx,
            score=round(float(score), 2),
            snippet=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        )
        sources.append(source)

    return ChatResponse(answer=cleaned, sources=sources)
