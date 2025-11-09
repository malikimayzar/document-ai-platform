from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import shutil
import os
import time
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
app_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(app_dir)

# ✅ Fixed import paths - sesuaikan dengan struktur folder Anda
try:
    from ml.ocr_engine import OCREngine
    from ml.nlp_engine import NLPEngine
    from backend.database import DatabaseManager
    from config import settings
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Fallback imports
    try:
        from app.ml.ocr_engine import OCREngine
        from app.ml.nlp_engine import NLPEngine  
        from app.backend.database import DatabaseManager
        from app.config import settings
    except ImportError:
        raise ImportError("Cannot import required modules. Check your project structure.")

# Initialize FastAPI
app = FastAPI(title="Document AI Platform API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Improved engine initialization with error handling
try:
    ocr_engine = OCREngine(engine=settings.OCR_ENGINE, lang=settings.OCR_LANG)
    nlp_engine = NLPEngine()
    db = DatabaseManager(settings.DATABASE_URL)
    logger.info("All engines initialized successfully")
except Exception as e:
    logger.error(f"Engine initialization failed: {e}")
    # Set to None to avoid runtime errors
    ocr_engine = None
    nlp_engine = None
    db = None

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Pydantic models
class QuestionRequest(BaseModel):
    document_id: int
    question: str

class SummaryRequest(BaseModel):
    document_id: int
    max_length: Optional[int] = 150

class SearchRequest(BaseModel):
    document_id: int
    query: str

# Health check dengan status engines
@app.get("/")
async def root():
    engine_status = {
        "ocr_engine": "ready" if ocr_engine else "failed",
        "nlp_engine": "ready" if nlp_engine else "failed", 
        "database": "ready" if db else "failed"
    }
    
    return {
        "message": "Document AI Platform API",
        "version": "1.0.0",
        "status": "running",
        "engines": engine_status
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Upload document dengan improvements
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    file_path = None  # Track file path for cleanup
    try:
        # ✅ Check if engines are ready
        if not ocr_engine or not db:
            raise HTTPException(status_code=503, detail="Service temporarily unavailable. Engines not ready.")
        
        # Validate file type
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        # Extract text
        start_time = time.time()
        extracted_text = ocr_engine.extract_text(str(file_path))
        processing_time = time.time() - start_time
        
        # Save to database
        doc_id = db.save_document(
            filename=file.filename,
            file_type=file_extension,
            file_size=file_size,
            extracted_text=extracted_text,
            processing_time=processing_time
        )
        
        return {
            "document_id": doc_id,
            "filename": file.filename,
            "file_size": file_size,
            "processing_time": round(processing_time, 2),
            "text_length": len(extracted_text),
            "preview": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # ✅ Cleanup uploaded file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup file {file_path}: {e}")

# Get document text
@app.get("/documents/{doc_id}")
async def get_document(doc_id: int):
    """Get document by ID"""
    try:
        if not db:
            raise HTTPException(status_code=503, detail="Database not available")
            
        doc = db.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "file_size": doc.file_size,
            "uploaded_at": doc.uploaded_at,
            "text": doc.extracted_text,
            "processing_time": doc.processing_time
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# List all documents
@app.get("/documents")
async def list_documents(limit: int = 50):
    """List all documents"""
    try:
        if not db:
            raise HTTPException(status_code=503, detail="Database not available")
            
        docs = db.get_all_documents(limit)
        return {
            "total": len(docs),
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "file_type": doc.file_type,
                    "uploaded_at": doc.uploaded_at,
                    "text_length": len(doc.extracted_text) if doc.extracted_text else 0
                }
                for doc in docs
            ]
        }
    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Question Answering dengan engine check
@app.post("/qa")
async def question_answering(request: QuestionRequest):
    """Answer questions about document"""
    try:
        if not nlp_engine or not db:
            raise HTTPException(status_code=503, detail="NLP engine not available")
            
        doc = db.get_document(request.document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        answer = nlp_engine.answer_question(request.question, doc.extracted_text)
        
        # Save analysis
        db.save_analysis(
            document_id=request.document_id,
            analysis_type="qa",
            query=request.question,
            result=answer
        )
        
        return answer
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"QA error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Summarization dengan engine check  
@app.post("/summarize")
async def summarize_document(request: SummaryRequest):
    """Generate document summary"""
    try:
        if not nlp_engine or not db:
            raise HTTPException(status_code=503, detail="NLP engine not available")
            
        doc = db.get_document(request.document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        summary = nlp_engine.summarize_text(doc.extracted_text, max_length=request.max_length)
        
        result = {"summary": summary}
        
        # Save analysis
        db.save_analysis(
            document_id=request.document_id,
            analysis_type="summary",
            result=result
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Entity Extraction dengan engine check
@app.post("/entities/{doc_id}")
async def extract_entities(doc_id: int):
    """Extract named entities from document"""
    try:
        if not nlp_engine or not db:
            raise HTTPException(status_code=503, detail="NLP engine not available")
            
        doc = db.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        entities = nlp_engine.extract_entities(doc.extracted_text)
        
        # Save analysis
        db.save_analysis(
            document_id=doc_id,
            analysis_type="entities",
            result=entities
        )
        
        return entities
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Sentiment Analysis dengan engine check
@app.post("/sentiment/{doc_id}")
async def analyze_sentiment(doc_id: int):
    """Analyze document sentiment"""
    try:
        if not nlp_engine or not db:
            raise HTTPException(status_code=503, detail="NLP engine not available")
            
        doc = db.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        sentiment = nlp_engine.analyze_sentiment(doc.extracted_text)
        
        # Save analysis
        db.save_analysis(
            document_id=doc_id,
            analysis_type="sentiment",
            result=sentiment
        )
        
        return sentiment
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search in document dengan engine check
@app.post("/search")
async def search_document(request: SearchRequest):
    """Search for text in document"""
    try:
        if not nlp_engine or not db:
            raise HTTPException(status_code=503, detail="NLP engine not available")
            
        doc = db.get_document(request.document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        results = nlp_engine.search_in_text(doc.extracted_text, request.query)
        
        return {
            "query": request.query,
            "total_results": len(results),
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get document analysis history
@app.get("/analyses/{doc_id}")
async def get_analyses(doc_id: int):
    """Get all analyses for a document"""
    try:
        if not db:
            raise HTTPException(status_code=503, detail="Database not available")
            
        analyses = db.get_document_analyses(doc_id)
        return {
            "document_id": doc_id,
            "total": len(analyses),
            "analyses": analyses
        }
    except Exception as e:
        logger.error(f"Get analyses error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Delete document
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    """Delete document and its analyses"""
    try:
        if not db:
            raise HTTPException(status_code=503, detail="Database not available")
            
        db.delete_document(doc_id)
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)