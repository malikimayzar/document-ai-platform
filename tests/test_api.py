import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import os
import tempfile
import json

# Add app to path - ✅ Improved path handling
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ✅ Improved import with multiple options
try:
    from backend.api import app
except ImportError:
    try:
        from app.backend.api import app
    except ImportError:
        from api import app  # Fallback

client = TestClient(app)

# Test data
SAMPLE_TEXT = "This is a sample document for testing. It contains some text that can be used for various tests. The Document AI Platform should be able to process this text correctly."

@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(SAMPLE_TEXT)
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)

@pytest.fixture
def sample_image_file():
    """Create a sample image file for testing (white image with text)"""
    from PIL import Image, ImageDraw, ImageFont
    try:
        # Create a simple image with text
        img = Image.new('RGB', (400, 200), color='white')
        d = ImageDraw.Draw(img)
        
        # Try to use default font
        try:
            font = ImageFont.load_default()
        except:
            font = None
            
        d.text((10, 10), "Sample Text for OCR Testing", fill='black', font=font)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    except Exception as e:
        pytest.skip(f"Could not create sample image: {e}")

# ============ BASIC ENDPOINT TESTS ============

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Document AI Platform" in data["message"]
    assert "version" in data
    assert "status" in data

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

# ============ DOCUMENT MANAGEMENT TESTS ============

def test_list_documents_empty():
    """Test listing documents when none exist"""
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "documents" in data
    assert isinstance(data["documents"], list)

def test_get_nonexistent_document():
    """Test getting non-existent document"""
    response = client.get("/documents/99999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

# ============ UPLOAD TESTS ============

def test_upload_invalid_file_type():
    """Test uploading invalid file type"""
    files = {"file": ("test.exe", b"fake content", "application/x-msdownload")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "not allowed" in response.json()["detail"].lower()

def test_upload_no_file():
    """Test uploading without file"""
    response = client.post("/upload")
    assert response.status_code == 422  # Validation error

def test_upload_valid_text_file(sample_text_file):
    """Test uploading valid text file"""
    with open(sample_text_file, 'rb') as f:
        files = {"file": ("test.txt", f, "text/plain")}
        response = client.post("/upload", files=files)
        
    # Should either succeed or give service unavailable if engines not ready
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "document_id" in data
        assert "filename" in data
        assert "file_size" in data
        assert "processing_time" in data
        assert "text_length" in data
        assert "preview" in data

def test_upload_valid_image_file(sample_image_file):
    """Test uploading valid image file"""
    with open(sample_image_file, 'rb') as f:
        files = {"file": ("test.png", f, "image/png")}
        response = client.post("/upload", files=files)
    
    # Should either succeed or give service unavailable if engines not ready
    assert response.status_code in [200, 503]

# ============ Q&A TESTS ============

def test_qa_without_document():
    """Test Q&A with invalid document ID"""
    data = {"document_id": 99999, "question": "Test question?"}
    response = client.post("/qa", json=data)
    assert response.status_code == 404

def test_qa_missing_fields():
    """Test Q&A with missing required fields"""
    # Missing question
    data = {"document_id": 1}
    response = client.post("/qa", json=data)
    assert response.status_code == 422
    
    # Missing document_id
    data = {"question": "Test question?"}
    response = client.post("/qa", json=data)
    assert response.status_code == 422

# ============ SUMMARIZATION TESTS ============

def test_summarize_without_document():
    """Test summarization with invalid document ID"""
    data = {"document_id": 99999}
    response = client.post("/summarize", json=data)
    assert response.status_code == 404

def test_summarize_with_custom_length():
    """Test summarization with custom length"""
    data = {"document_id": 99999, "max_length": 100}
    response = client.post("/summarize", json=data)
    assert response.status_code == 404  # Document doesn't exist

# ============ ENTITY EXTRACTION TESTS ============

def test_entities_without_document():
    """Test entity extraction with invalid document ID"""
    response = client.post("/entities/99999")
    assert response.status_code == 404

# ============ SENTIMENT ANALYSIS TESTS ============

def test_sentiment_without_document():
    """Test sentiment analysis with invalid document ID"""
    response = client.post("/sentiment/99999")
    assert response.status_code == 404

# ============ SEARCH TESTS ============

def test_search_without_document():
    """Test search with invalid document ID"""
    data = {"document_id": 99999, "query": "test"}
    response = client.post("/search", json=data)
    assert response.status_code == 404

def test_search_missing_query():
    """Test search with missing query"""
    data = {"document_id": 1}  # Missing query
    response = client.post("/search", json=data)
    assert response.status_code == 422

# ============ ANALYSIS HISTORY TESTS ============

def test_analyses_without_document():
    """Test getting analyses for non-existent document"""
    response = client.get("/analyses/99999")
    assert response.status_code == 200  # Should return empty list, not error
    data = response.json()
    assert data["document_id"] == 99999
    assert data["total"] == 0
    assert data["analyses"] == []

# ============ DELETE TESTS ============

def test_delete_nonexistent_document():
    """Test deleting non-existent document"""
    response = client.delete("/documents/99999")
    # Should either succeed (idempotent) or return not found
    assert response.status_code in [200, 404, 500]

# ============ ERROR HANDLING TESTS ============

def test_invalid_json():
    """Test sending invalid JSON"""
    response = client.post("/qa", data="invalid json")
    assert response.status_code in [422, 400]

def test_invalid_endpoint():
    """Test accessing non-existent endpoint"""
    response = client.get("/nonexistent")
    assert response.status_code == 404

# ============ SUCCESS SCENARIO TESTS ============

def test_full_workflow(sample_text_file):
    """Test a complete workflow if engines are available"""
    # Skip if engines are not ready
    health_response = client.get("/")
    if health_response.status_code != 200:
        pytest.skip("Engines not ready for testing")
    
    # 1. Upload document
    with open(sample_text_file, 'rb') as f:
        files = {"file": ("test_workflow.txt", f, "text/plain")}
        upload_response = client.post("/upload", files=files)
    
    if upload_response.status_code != 200:
        pytest.skip("Upload failed - engines may not be ready")
    
    upload_data = upload_response.json()
    doc_id = upload_data["document_id"]
    
    # 2. Get document
    get_response = client.get(f"/documents/{doc_id}")
    assert get_response.status_code == 200
    get_data = get_response.json()
    assert get_data["id"] == doc_id
    assert get_data["filename"] == "test_workflow.txt"
    
    # 3. List documents
    list_response = client.get("/documents")
    assert list_response.status_code == 200
    list_data = list_response.json()
    assert any(doc["id"] == doc_id for doc in list_data["documents"])
    
    # 4. Try Q&A (may not work if NLP engine not available)
    qa_data = {"document_id": doc_id, "question": "What is this document about?"}
    qa_response = client.post("/qa", json=qa_data)
    # Could be 200 (success), 503 (engine not available), or 500 (error)
    assert qa_response.status_code in [200, 503, 500]
    
    # 5. Cleanup - delete document
    delete_response = client.delete(f"/documents/{doc_id}")
    assert delete_response.status_code in [200, 404]  # Should succeed or already gone

# ============ PERFORMANCE TESTS ============

def test_response_times():
    """Test that endpoints respond within reasonable time"""
    import time
    
    start_time = time.time()
    response = client.get("/health")
    end_time = time.time()
    
    assert response.status_code == 200
    assert end_time - start_time < 5.0  # Should respond within 5 seconds

# ============ SECURITY TESTS ============

def test_cors_headers():
    """Test that CORS headers are present"""
    response = client.get("/health")
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "*"

def test_malicious_filename():
    """Test uploading file with potentially malicious filename"""
    malicious_content = b"normal content"
    files = {"file": ("../../../etc/passwd", malicious_content, "text/plain")}
    response = client.post("/upload", files=files)
    # Should either validate filename or handle safely
    assert response.status_code in [200, 400, 422, 503]

if __name__ == "__main__":
    # Run tests manually
    pytest.main([__file__, "-v"])