import os
import uuid
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
import tempfile
import logging

from src.document_processing.doc_processor import DocumentProcessor
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_database.milvus_vector_db import MilvusVectorDB
from src.generation.rag import RAGGenerator
from src.web_scraping.web_scraper import WebScraper

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Research Assistant API",
    description="Backend API for RAG pipeline",
    version="1.0.0"
)

# Global instances (simplified for Assessment)
db_path = "./milvus_backend.db"
collection_name = "backend_collection"

try:
    doc_processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator()
    vector_db = MilvusVectorDB(db_path=db_path, collection_name=collection_name)
    openai_key = os.getenv("OPENAI_API_KEY")
    rag_generator = RAGGenerator(
        embedding_generator=embedding_generator,
        vector_db=vector_db,
        openai_api_key=openai_key
    )
    web_scraper = WebScraper()

    # Ensure index exists if the collection was just created
    try:
        vector_db.create_index(use_binary_quantization=False)
    except Exception as e:
        logger.info(f"Index might already exist or skipped: {e}")
        
except Exception as init_err:
    logger.error(f"Failed to initialize core components: {init_err}")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "AI Research Assistant"}

@app.post("/ingest")
def ingest_data(file: Optional[UploadFile] = File(None), url: Optional[str] = Form(None)):
    """
    Ingests data from a file upload (PDF/TXT) or a web URL.
    Provide either `file` as multipart/form-data or `url` as a text field.
    """
    results = []
    
    if url:
        try:
            chunks = web_scraper.scrape_url(url)
            if not chunks:
                results.append({"url": url, "status": "failed", "detail": "No content extracted. The site might be blocking basic bots."})
            else:
                for chunk in chunks:
                    chunk.source_file = url
                embedded_chunks = embedding_generator.generate_embeddings(chunks)
                vector_db.insert_embeddings(embedded_chunks)
                results.append({"url": url, "status": "success", "chunks_processed": len(chunks)})
        except Exception as e:
            results.append({"url": url, "status": "failed", "detail": str(e)})
            
    if file:
        try:
            suffix = f".{file.filename.split('.')[-1]}" if '.' in file.filename else ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                content = file.file.read()
                tmp_file.write(content)
                temp_path = tmp_file.name

            chunks = doc_processor.process_document(temp_path)
            for chunk in chunks:
                chunk.source_file = file.filename
                
            embedded_chunks = embedding_generator.generate_embeddings(chunks)
            vector_db.insert_embeddings(embedded_chunks)
            os.unlink(temp_path)
            results.append({"filename": file.filename, "status": "success", "chunks_processed": len(chunks)})
        except Exception as e:
            if 'temp_path' in locals():
                os.unlink(temp_path)
            results.append({"filename": file.filename, "status": "failed", "detail": str(e)})
            
    if not results:
        raise HTTPException(status_code=400, detail="Must provide either 'file' or 'url'")
        
    return {"results": results}

@app.post("/query", response_model=QueryResponse)
def query_system(request: QueryRequest):
    """
    Queries the RAG system using the provided question.
    """
    try:
        result = rag_generator.generate_response(request.query)
        sources_info = []
        for source in result.sources_used:
            sources_info.append(source)
            
        return QueryResponse(answer=result.response, sources=sources_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Trigger reload over again
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
