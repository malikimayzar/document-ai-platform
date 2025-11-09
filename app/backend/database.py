from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json 

Base = declarative_base()

class Document(Base):
    """Documents model"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50))
    file_size = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    extracted_text = Column(Text)
    processing_time = Column(Float)

class Analysis(Base):
    """Analysis result model"""  # ✅ Fixed typo
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, nullable=False)
    analysis_type = Column(String(50))
    query = Column(Text)
    result = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Database manager for the application"""

    def __init__(self, database_url: str = "sqlite:///./documents.db"):
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def save_document(self, filename: str, file_type: str, file_size: int,
                      extracted_text: str, processing_time: float) -> int:
        """Save documents to database"""
        session = self.get_session()
        try:
            doc = Document(
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                extracted_text=extracted_text,
                processing_time=processing_time
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)
            return doc.id
        finally:
            session.close()

    def save_analysis(self, document_id: int, analysis_type: str, 
                      result: dict, query: str = None) -> int:
        """Save analysis result"""
        session = self.get_session()
        try:
            analysis = Analysis(
                document_id=document_id,
                analysis_type=analysis_type,
                query=query,
                result=json.dumps(result)
            )
            session.add(analysis)
            session.commit()
            session.refresh(analysis)
            return analysis.id
        finally:
            session.close()

    def get_document(self, doc_id: int) -> Document:
        """Get documents by ID"""
        session = self.get_session()        
        try:
            return session.query(Document).filter(Document.id == doc_id).first()
        finally:
            session.close()

    def get_all_documents(self, limit: int = 50):
        """Get all documents"""
        session = self.get_session()
        try:
            return session.query(Document).order_by(Document.uploaded_at.desc()).limit(limit).all()
        finally:
            session.close()

    def get_document_analyses(self, doc_id: int):
        """Get all analyses for a document"""
        session = self.get_session()
        try:
            analyses = session.query(Analysis).filter(Analysis.document_id == doc_id).all()
            return [
                {
                    "id": a.id,
                    "type": a.analysis_type,
                    "query": a.query,
                    "result": json.loads(a.result),  # ✅ Fixed: json.loads() bukan json.load()
                    "created_at": a.created_at
                }
                for a in analyses
            ]
        finally:
            session.close()

    def delete_document(self, doc_id: int):
        """Delete document and its analyses"""
        session = self.get_session()
        try:
            # ✅ Fixed both typos
            session.query(Analysis).filter(Analysis.document_id == doc_id).delete()
            session.query(Document).filter(Document.id == doc_id).delete()
            session.commit()
        finally:
            session.close()