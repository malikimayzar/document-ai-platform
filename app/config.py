"""Application configuration"""
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    APP_NAME: str = "Document AI Platform"
    VERSION: str = "1.0.0"
    
    # Model settings
    NLP_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    QA_MODEL: str = "deepset/roberta-base-squad2"
    SUMMARIZATION_MODEL: str = "facebook/bart-large-cnn"
    
    # OCR settings
    OCR_ENGINE: str = "tesseract"
    OCR_LANG: str = "eng"
    
    # Database
    DATABASE_URL: str = "sqlite:///./documents.db"
    
    # File upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: list = [".pdf", ".png", ".jpg", ".jpeg", ".txt"]
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings()
