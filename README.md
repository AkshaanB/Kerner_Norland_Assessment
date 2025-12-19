# EduProvider Institute Chatbot API

A RAG (Retrieval-Augmented Generation) based chatbot API for EduProvider Institute, built with FastAPI and LangChain.

## Features

- FastAPI-based REST API
- RAG implementation using LangChain
- MMR (Maximal Marginal Relevance) retrieval to reduce duplicate chunks
- Vector store with ChromaDB
- Ollama integration for embeddings and LLM
- Document loading and processing
- Context-aware question answering with source citations
- Structured JSON responses with answer and sources

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Required Ollama models:
  - `mxbai-embed-large` (for embeddings)
  - `deepseek-r1:1.5b` (for LLM)

## Installation

Install required packages:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
Create a `.env` file in the root directory with the following:
```env
EMBEDDING_MODEL_NAME=mxbai-embed-large
MODEL_NAME=deepseek-r1:1.5b
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=4
FETCH_K=20
LAMBDA_MULT=0.5
```

5. Ensure Ollama is running and has the required models:
```bash
ollama pull mxbai-embed-large
ollama pull deepseek-r1:1.5b
```

## Running the API

### Option 1: Using uvicorn directly
```bash
uvicorn routes.chat_api:app --reload
```

### Option 2: Specify host and port
```bash
uvicorn routes.chat_api:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: For production (without reload)
```bash
uvicorn routes.chat_api:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- Local: http://localhost:8000
- Network: http://0.0.0.0:8000 (if using --host 0.0.0.0)

## API Documentation

Once the server is running, you can access:
- Interactive API docs (Swagger UI): http://localhost:8000/docs
- Alternative API docs (ReDoc): http://localhost:8000/redoc

## API Endpoints

### 1. Root Endpoint
**GET /**
```bash
curl http://localhost:8000/
```
Response:
```json
{
  "message": "Welcome to the EduProvider Institute Chatbot API."
}
```

### 2. Chat Endpoint
**POST /chat**

Send a question to the chatbot and get an answer based on the document context.

Request body:
```json
{
  "question": "What courses does EduProvider offer?"
}
```

Example using curl:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What courses does EduProvider offer?\"}"
```

Example using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"question": "What courses does EduProvider offer?"}
)
print(response.json())
```

Example Response:
```json
{
  "answer": "EduProvider offers courses in Computer Science, Data Science, and Business Administration.",
  "sources": [
    {
      "doc": "data/courses.txt",
      "chunk_id": 0,
      "score": 0.85,
      "snippet": "Our comprehensive course catalog includes Computer Science with specializations in AI and Software Engineering, Data Science covering machine learning and analytics..."
    },
    {
      "doc": "data/programs.txt",
      "chunk_id": 1,
      "score": 0.72,
      "snippet": "Business Administration program offers concentrations in Management, Finance, and Marketing. Students can complete the program in 2-4 years..."
    }
  ]
}
```

## Project Structure

```
Kerner_Norland/
----- routes/
      ----- chat_api.py          # FastAPI application and endpoints
----- services/
      ----- load_docs.py         # Document loading functionality
      ----- split_docs.py        # Document splitting/chunking
      ----- rag.py               # RAG implementation
      ----- clean.py             # Response cleaning utilities
      ----- vector_store.py      # Vector store management
----- data/                    # Document storage directory
----- chromadb/                # ChromaDB vector store (auto-generated)
----- .env                     # Environment configuration
----- requirements.txt         # Python dependencies
----- test.py                  # Test script
----- README.md               # This file
```

## How It Works

1. **Document Loading**: Documents are loaded from the `data/` directory
2. **Vector Store**: Documents are embedded using Ollama's `mxbai-embed-large` model and stored in ChromaDB
3. **RAG Pipeline**: When a question is received:
   - The question is embedded
   - MMR (Maximal Marginal Relevance) retrieves diverse, relevant chunks to reduce duplicates
   - Top K document chunks are selected balancing relevance and diversity
   - The LLM generates an answer based on the context
   - The response is cleaned and returned with source citations

## Configuration

Edit `.env` file to customize:
- `EMBEDDING_MODEL_NAME`: Ollama embedding model
- `MODEL_NAME`: Ollama LLM model
- `CHUNK_SIZE`: Document chunk size for splitting
- `CHUNK_OVERLAP`: Overlap between chunks
- `TOP_K`: Number of relevant chunks to retrieve
- `FETCH_K`: Number of candidate chunks to fetch for MMR (default: 20)
- `LAMBDA_MULT`: MMR diversity parameter (0=max diversity, 1=max relevance, default: 0.5)

## Testing

Run the test script to verify the setup:
```bash
python test.py
```

## Troubleshooting

### Import Errors
If you encounter import errors with langchain, ensure you have installed:
```bash
pip install langchain-classic langchain-core langchain-community langchain-ollama
```

### Ollama Connection Issues
Ensure Ollama is running:
```bash
ollama serve
```

### Port Already in Use
If port 8000 is already in use, specify a different port:
```bash
uvicorn routes.chat_api:app --port 8001
```

## Development

For development with auto-reload:
```bash
uvicorn routes.chat_api:app --reload --log-level debug
```

