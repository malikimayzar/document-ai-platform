import streamlit as st
st.set_page_config(
    page_title="Document AI Platform",
    page_icon="📄",
    layout="wide"
)

import pytesseract
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import time
import io
import tempfile
import os
from collections import Counter

tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print (f"Tesseract configured at: {tesseract_path}")
else:
    st.error(f" Tesseract not found at: {tesseract_path}")
    st.info("Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")


# Try to import ML libraries (with fallback)
try:
    from transformers import pipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("⚠️ Transformers not installed. Some features will be limited.")

# Try to import PDF libraries
try:
    from pdf2image import convert_from_bytes
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ pdf2image not installed. PDF processing will be limited.")

# Comprehensive session state
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'qa_model' not in st.session_state:
    st.session_state.qa_model = None
if 'current_filename' not in st.session_state:
    st.session_state.current_filename = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# ============ OCR FUNCTIONS ============

def preprocess_image(image):
    """Preprocess image for better OCR"""
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply thresholding
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        return thresh
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return image

def extract_text_from_image(image):
    """Extract text from image using Tesseract"""
    try:
        processed = preprocess_image(image)
        text = pytesseract.image_to_string(processed, lang='eng')
        return text.strip()
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    if not PDF_AVAILABLE:
        return "Error: pdf2image not installed. Please install with: pip install pdf2image"
    
    temp_file_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            temp_file_path = tmp_file.name
        
        # Convert PDF to images
        images = convert_from_bytes(open(temp_file_path, 'rb').read())
        
        text = ""
        for i, image in enumerate(images):
            with st.expander(f"Page {i+1}", expanded=False):
                st.image(image, caption=f"Page {i+1}", use_column_width=True)
            
            page_text = extract_text_from_image(image)
            text += f"\n--- Page {i+1} ---\n{page_text}\n"
        
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"
    finally:
        # Cleanup temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                st.warning(f"Could not cleanup temporary file: {e}")

def extract_text_from_txt(txt_file):
    """Extract text from TXT file"""
    try:
        text = txt_file.read().decode('utf-8')
        return text.strip()
    except Exception as e:
        return f"Error reading text file: {str(e)}"

# ============ NLP FUNCTIONS ============

@st.cache_resource
def load_qa_model():
    """Load Q&A model (cached)"""
    if not ML_AVAILABLE:
        return None
    try:
        return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    except Exception as e:
        st.error(f"Error loading Q&A model: {e}")
        return None

def answer_question(question, context):
    """Answer question based on context"""
    if st.session_state.qa_model is None:
        st.session_state.qa_model = load_qa_model()
    
    if st.session_state.qa_model is None:
        return {"answer": "Q&A model not available", "confidence": 0.0}
    
    try:
        # Truncate context if too long
        max_length = 512
        if len(context.split()) > max_length:
            words = context.split()[:max_length]
            context = " ".join(words)
            st.warning(f"⚠️ Context truncated to {max_length} words for better performance")
        
        result = st.session_state.qa_model(question=question, context=context)
        return {
            "answer": result["answer"],
            "confidence": round(result["score"], 4)
        }
    except Exception as e:
        return {"answer": f"Error processing question: {str(e)}", "confidence": 0.0}

def analyze_document_stats(text):
    """Analyze document statistics"""
    stats = {}
    
    # Basic stats
    stats['characters'] = len(text)
    stats['words'] = len(text.split())
    stats['lines'] = len(text.split('\n'))
    stats['paragraphs'] = len([p for p in text.split('\n\n') if p.strip()])
    stats['sentences'] = len([s for s in text.split('.') if s.strip()])
    
    # Word frequency
    words = text.lower().split()
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'are', 'were', 'this', 'that', 'it', 'as', 'be', 'been', 'have', 'has', 'had'}
    filtered_words = [w for w in words if w not in stop_words and len(w) > 3 and w.isalpha()]
    stats['word_freq'] = Counter(filtered_words).most_common(15)
    
    # Reading time estimate (average 200 words per minute)
    stats['reading_time'] = max(1, round(len(words) / 200))
    
    return stats

# ============ MAIN APP ============

# Header
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .feature-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📄 Document AI Platform</div>', unsafe_allow_html=True)
st.markdown("### Extract, Analyze, and Understand Your Documents")

# Sidebar
with st.sidebar:
    st.header("⚙️ Features")
    
    feature = st.radio(
        "Select Feature:",
        ["📤 Upload & Extract", "💬 Q&A", "📊 Analysis", "ℹ️ About"]
    )
    
    st.markdown("---")
    
    # Current document info
    if st.session_state.processing_complete:
        st.success("📄 Document Loaded")
        if st.session_state.current_filename:
            st.write(f"**File:** {st.session_state.current_filename}")
        if st.session_state.extracted_text:
            st.write(f"**Text Length:** {len(st.session_state.extracted_text)} chars")
            st.write(f"**Words:** {len(st.session_state.extracted_text.split())}")
    
    # System check
    with st.expander("🔍 System Check"):
        try:
            version = pytesseract.get_tesseract_version()
            st.success(f"✅ Tesseract: {version}")
        except:
            st.error("❌ Tesseract not found")
        
        if ML_AVAILABLE:
            st.success("✅ Transformers installed")
        else:
            st.warning("⚠️ Transformers not installed")
            
        if PDF_AVAILABLE:
            st.success("✅ PDF support available")
        else:
            st.warning("⚠️ PDF support not available")

# Main content
if feature == "📤 Upload & Extract":
    st.header("📤 Upload Document")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'pdf', 'png', 'jpg', 'jpeg'],
            help="Upload PDF, Image, or Text file"
        )
        
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1].lower()
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📊 File size: {uploaded_file.size / 1024:.2f} KB")
            
            # Show preview for images
            if file_type in ['png', 'jpg', 'jpeg']:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("🚀 Extract Text", type="primary"):
                with st.spinner("Processing..."):
                    start_time = time.time()
                    
                    # Extract text based on file type
                    try:
                        if file_type == 'txt':
                            text = extract_text_from_txt(uploaded_file)
                        elif file_type == 'pdf':
                            if not PDF_AVAILABLE:
                                st.error("PDF processing not available. Install pdf2image: pip install pdf2image")
                                text = None
                            else:
                                text = extract_text_from_pdf(uploaded_file)
                        elif file_type in ['png', 'jpg', 'jpeg']:
                            image = Image.open(uploaded_file)
                            text = extract_text_from_image(image)
                        else:
                            text = "Unsupported file type"
                    
                        processing_time = time.time() - start_time
                        
                        if text and not text.startswith("Error"):
                            # Store in session
                            st.session_state.extracted_text = text
                            st.session_state.current_filename = uploaded_file.name
                            st.session_state.processing_complete = True
                            
                            # Show results
                            st.success(f"✅ Text extracted in {processing_time:.2f} seconds!")
                            
                            # Metrics
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Characters", len(text))
                            col_b.metric("Words", len(text.split()))
                            col_c.metric("Processing Time", f"{processing_time:.2f}s")
                            
                            # Preview
                            st.subheader("📄 Extracted Text Preview")
                            with st.expander("Click to view full text", expanded=False):
                                st.text_area("Extracted Text", text, height=300, key="text_preview")
                        else:
                            st.error(f"❌ {text}")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing file: {str(e)}")
    
    with col2:
        st.markdown("### 💡 Tips")
        st.info("""
        **For best results:**
        - Use clear, high-resolution images
        - Ensure good lighting
        - Keep text horizontal
        - Remove shadows and glare
        
        **Supported formats:**
        - Images: PNG, JPG, JPEG
        - Documents: PDF, TXT
        """)
        
        st.markdown("### 🔧 Requirements")
        if not PDF_AVAILABLE:
            st.code("pip install pdf2image", language="bash")
        if not ML_AVAILABLE:
            st.code("pip install transformers", language="bash")

elif feature == "💬 Q&A":
    st.header("💬 Question & Answer")
    
    if not st.session_state.processing_complete or st.session_state.extracted_text is None:
        st.warning("⚠️ Please upload and extract text first!")
        if st.button("Go to Upload"):
            # Simulate navigation
            st.session_state.feature = "📤 Upload & Extract"
            st.rerun()
    else:
        st.success("📄 Document loaded!")
        
        # Show text preview
        with st.expander("View document text preview"):
            preview_text = st.session_state.extracted_text[:1000] + "..." if len(st.session_state.extracted_text) > 1000 else st.session_state.extracted_text
            st.text(preview_text)
            st.caption(f"Full text: {len(st.session_state.extracted_text)} characters")
        
        question = st.text_input("Ask a question about your document:", placeholder="e.g., What is the main topic of this document?")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔍 Get Answer", type="primary", use_container_width=True):
                if question:
                    if not ML_AVAILABLE:
                        st.error("❌ Transformers library not installed. Install with: pip install transformers")
                    else:
                        with st.spinner("Finding answer..."):
                            result = answer_question(question, st.session_state.extracted_text)
                            
                            st.subheader("💡 Answer")
                            st.info(result["answer"])
                            
                            st.metric("Confidence", f"{result['confidence']:.2%}")
                            
                            if result["confidence"] < 0.3:
                                st.warning("⚠️ Low confidence. The answer might not be accurate.")
                else:
                    st.error("Please enter a question!")
        
        with col2:
            if st.button("🔄 Clear", use_container_width=True):
                st.rerun()

elif feature == "📊 Analysis":
    st.header("📊 Document Analysis")
    
    if not st.session_state.processing_complete or st.session_state.extracted_text is None:
        st.warning("⚠️ Please upload and extract text first!")
    else:
        text = st.session_state.extracted_text
        
        # Basic statistics
        st.subheader("📈 Document Statistics")
        stats = analyze_document_stats(text)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Characters", stats['characters'])
        col2.metric("Words", stats['words'])
        col3.metric("Sentences", stats['sentences'])
        col4.metric("Reading Time", f"{stats['reading_time']} min")
        
        col5, col6, col7 = st.columns(3)
        col5.metric("Lines", stats['lines'])
        col6.metric("Paragraphs", stats['paragraphs'])
        col7.metric("Avg Word Length", f"{stats['characters']/max(1, stats['words']):.1f}")
        
        # Word frequency
        st.subheader("🔤 Most Common Words")
        if stats['word_freq']:
            col8, col9 = st.columns(2)
            
            with col8:
                for word, count in stats['word_freq'][:8]:
                    st.write(f"**{word}**: {count} times")
            
            with col9:
                # Simple bar chart using text
                for word, count in stats['word_freq'][8:]:
                    st.write(f"**{word}**: {count} times")
        else:
            st.info("Not enough unique words for frequency analysis")
        
        # Search functionality
        st.subheader("🔍 Search in Document")
        search_query = st.text_input("Enter search term:")
        
        if search_query:
            occurrences = text.lower().count(search_query.lower())
            if occurrences > 0:
                st.success(f"Found **{occurrences}** occurrences of '{search_query}'")
                
                # Show context
                lines = text.split('\n')
                matches = [line for line in lines if search_query.lower() in line.lower()]
                
                st.write("**Matching lines:**")
                for i, match in enumerate(matches[:8]):  # Show first 8
                    highlighted = match.replace(search_query, f"**{search_query}**")
                    st.write(f"{i+1}. {highlighted}")
            else:
                st.warning(f"No occurrences found for '{search_query}'")

else:  # About
    st.header("ℹ️ About")
    
    st.markdown("""
    <div class="feature-card">
    ## 📄 Document AI Platform
    
    ### 🚀 Features
    - **OCR Text Extraction**: Extract text from images and PDFs using Tesseract
    - **Smart Q&A**: Ask questions about your documents using AI
    - **Document Analysis**: Get comprehensive insights and statistics
    
    ### 🔧 Technology Stack
    - **OCR**: Tesseract OCR Engine
    - **AI/ML**: Transformers (DistilBERT for Q&A)
    - **Image Processing**: OpenCV, PIL
    - **UI**: Streamlit
    - **PDF Processing**: pdf2image
    
    ### 📖 How to Use
    1. **Upload** your document (PDF, Image, or Text)
    2. **Extract** text using OCR
    3. **Analyze** content or ask questions
    
    ### ⚙️ Requirements
    - Python 3.8+
    - Tesseract OCR installed on system
    - Required Python packages
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🛠️ Installation")
    st.code("""
# Install Python packages
pip install streamlit pytesseract pillow opencv-python transformers torch pdf2image

# Install Tesseract OCR
# On Windows: download from https://github.com/UB-Mannheim/tesseract/wiki
# On Mac: brew install tesseract
# On Linux: sudo apt-get install tesseract-ocr
""", language="bash")
    
    st.markdown("### 🚀 Running the App")
    st.code("streamlit run app.py", language="bash")
    
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit & AI | Document AI Platform v2.0")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        👨‍💻 Dibuat oleh <b>malikimayzar</b><br><br>
        <a href="https://github.com/malikimayzar" target="_blank" style="margin-right: 15px;">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" 
            width="40" height="40">
        </a>
        <a href="https://instagram.com/malikimayzar" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png" 
            width="40" height="40">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
