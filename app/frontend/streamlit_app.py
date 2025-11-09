import streamlit as st
import requests
import time
from pathlib import Path
import sys
import os
import tempfile

# Add app directory to path - ✅ Improved path handling
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ✅ Improved import with error handling
try:
    from ml.ocr_engine import OCREngine
    from ml.nlp_engine import NLPEngine
    from backend.database import DatabaseManager
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.info("Please check your project structure and ensure all modules are available")

# Page config
st.set_page_config(
    page_title="Document AI Platform",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .document-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_doc_id' not in st.session_state:
    st.session_state.current_doc_id = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'current_filename' not in st.session_state:
    st.session_state.current_filename = None
if 'ocr_engine' not in st.session_state:
    st.session_state.ocr_engine = None
if 'nlp_engine' not in st.session_state:
    st.session_state.nlp_engine = None
if 'db' not in st.session_state:
    st.session_state.db = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Initialize engines (cached) dengan error handling
@st.cache_resource
def load_ocr_engine():
    try:
        return OCREngine(engine="tesseract", lang="eng")
    except Exception as e:
        st.error(f"❌ Failed to load OCR engine: {e}")
        return None

@st.cache_resource
def load_nlp_engine():
    try:
        return NLPEngine()
    except Exception as e:
        st.error(f"❌ Failed to load NLP engine: {e}")
        return None

@st.cache_resource
def load_database():
    try:
        return DatabaseManager()
    except Exception as e:
        st.error(f"❌ Failed to load database: {e}")
        return None

# Header
st.markdown('<div class="main-header">📄 Intelligent Document Analysis Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Extract, Analyze, and Understand Your Documents with AI</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Feature selection
    st.subheader("Available Features")
    feature = st.radio(
        "Select Analysis Type:",
        ["📤 Upload Document", "💬 Q&A", "📝 Summarize", "🏷️ Entity Extraction", 
         "😊 Sentiment Analysis", "🔍 Search", "📚 Document History"]
    )
    
    st.markdown("---")
    
    # Current document info
    if st.session_state.current_doc_id:
        st.success("📄 Document Loaded")
        st.write(f"**ID:** {st.session_state.current_doc_id}")
        if st.session_state.current_filename:
            st.write(f"**File:** {st.session_state.current_filename}")
        if st.session_state.extracted_text:
            st.write(f"**Text Length:** {len(st.session_state.extracted_text)} chars")
    
    # Info
    st.info("""
    **How to use:**
    1. Upload your document
    2. Choose an analysis type
    3. Get AI-powered insights!
    
    **Supported formats:**
    - PDF, PNG, JPG, JPEG, TXT
    """)
    
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit & Transformers")

# Main content area
if feature == "📤 Upload Document":
    st.header("📤 Upload Document")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'png', 'jpg', 'jpeg', 'txt'],
            help="Upload PDF, Image, or Text file"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📊 File size: {uploaded_file.size / 1024:.2f} KB")
            
            # Process button
            if st.button("🚀 Process Document", type="primary"):
                with st.spinner("Processing document... This may take a moment"):
                    temp_file_path = None
                    try:
                        # Initialize engines if not already done
                        if st.session_state.ocr_engine is None:
                            st.session_state.ocr_engine = load_ocr_engine()
                        if st.session_state.nlp_engine is None:
                            st.session_state.nlp_engine = load_nlp_engine()
                        if st.session_state.db is None:
                            st.session_state.db = load_database()
                        
                        # ✅ Check if engines loaded successfully
                        if st.session_state.ocr_engine is None:
                            st.error("❌ OCR engine failed to load. Please check installation.")
                            st.stop()
                        if st.session_state.db is None:
                            st.error("❌ Database failed to load. Please check configuration.")
                            st.stop()
                        
                        # Save file temporarily dengan cleanup
                        upload_dir = Path("uploads")
                        upload_dir.mkdir(exist_ok=True)
                        temp_file_path = upload_dir / uploaded_file.name
                        
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Extract text
                        start_time = time.time()
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Extracting text from document...")
                        progress_bar.progress(30)
                        
                        # Handle different file types
                        file_extension = Path(uploaded_file.name).suffix.lower()
                        if file_extension == '.txt':
                            with open(temp_file_path, 'r', encoding='utf-8') as f:
                                extracted_text = f.read()
                        else:
                            extracted_text = st.session_state.ocr_engine.extract_text(str(temp_file_path))
                        
                        progress_bar.progress(70)
                        status_text.text("Saving to database...")
                        
                        processing_time = time.time() - start_time
                        
                        # Save to database
                        doc_id = st.session_state.db.save_document(
                            filename=uploaded_file.name,
                            file_type=file_extension,
                            file_size=uploaded_file.size,
                            extracted_text=extracted_text,
                            processing_time=processing_time
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Processing complete!")
                        
                        # Store in session
                        st.session_state.current_doc_id = doc_id
                        st.session_state.extracted_text = extracted_text
                        st.session_state.current_filename = uploaded_file.name
                        st.session_state.processing_complete = True
                        
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Show results
                        st.success(f"✅ Document processed successfully in {processing_time:.2f} seconds!")
                        
                        # Display metrics
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Document ID", doc_id)
                        col_b.metric("Characters", len(extracted_text))
                        col_c.metric("Words", len(extracted_text.split()))
                        
                        # Show preview
                        st.subheader("📄 Extracted Text Preview")
                        with st.expander("Click to view full text", expanded=False):
                            st.text_area("Extracted Text", extracted_text, height=300, key="extracted_text_area")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")
                        st.info("Please check if all dependencies are installed and try again.")
                    finally:
                        # ✅ Cleanup temporary file
                        if temp_file_path and os.path.exists(temp_file_path):
                            try:
                                os.remove(temp_file_path)
                            except Exception as e:
                                st.warning(f"⚠️ Could not cleanup temporary file: {e}")
    
    with col2:
        st.markdown("### 📊 Stats")
        if st.session_state.db is None:
            st.session_state.db = load_database()
        
        if st.session_state.db:
            try:
                docs = st.session_state.db.get_all_documents(limit=10)
                st.metric("Total Documents", len(docs))
                
                if docs:
                    st.markdown("### Recent Documents")
                    for doc in docs[:5]:
                        with st.container():
                            st.markdown(f'<div class="document-card">', unsafe_allow_html=True)
                            st.write(f"**{doc.filename}**")
                            st.caption(f"ID: {doc.id} | {doc.uploaded_at.strftime('%Y-%m-%d')}")
                            st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error loading documents: {e}")

elif feature == "💬 Q&A":
    st.header("💬 Question & Answer")
    
    if not st.session_state.processing_complete or st.session_state.current_doc_id is None:
        st.warning("⚠️ Please upload and process a document first!")
        if st.button("Go to Upload"):
            st.session_state.feature = "📤 Upload Document"
            st.rerun()
    else:
        st.success(f"📄 Working with: {st.session_state.current_filename or 'Document'} (ID: {st.session_state.current_doc_id})")
        
        question = st.text_input("Ask a question about your document:", placeholder="e.g., What is the main topic?")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔍 Get Answer", type="primary", use_container_width=True):
                if question:
                    with st.spinner("Searching for answer..."):
                        try:
                            if st.session_state.nlp_engine is None:
                                st.session_state.nlp_engine = load_nlp_engine()
                            
                            if st.session_state.nlp_engine is None:
                                st.error("❌ NLP engine not available")
                                st.stop()
                            
                            answer = st.session_state.nlp_engine.answer_question(
                                question,
                                st.session_state.extracted_text
                            )
                            
                            # Display answer
                            st.subheader("💡 Answer")
                            st.info(answer.get("answer", "No answer found"))
                            
                            # Display confidence
                            confidence = answer.get("confidence", 0)
                            st.metric("Confidence Score", f"{confidence:.2%}")
                            
                            if confidence < 0.3:
                                st.warning("⚠️ Low confidence answer. The information might not be accurate.")
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.error("Please enter a question!")
        
        with col2:
            if st.button("🔄 Clear", use_container_width=True):
                st.rerun()

# ... (similar improvements for other features - keeping response concise)

elif feature == "📝 Summarize":
    st.header("📝 Document Summarization")
    
    if not st.session_state.processing_complete or st.session_state.current_doc_id is None:
        st.warning("⚠️ Please upload and process a document first!")
    else:
        st.success(f"📄 Working with: {st.session_state.current_filename or 'Document'} (ID: {st.session_state.current_doc_id})")
        
        max_length = st.slider("Summary length (words)", 50, 300, 150)
        
        if st.button("✨ Generate Summary", type="primary"):
            with st.spinner("Generating summary..."):
                try:
                    if st.session_state.nlp_engine is None:
                        st.session_state.nlp_engine = load_nlp_engine()
                    
                    if st.session_state.nlp_engine is None:
                        st.error("❌ NLP engine not available")
                        st.stop()
                    
                    summary = st.session_state.nlp_engine.summarize_text(
                        st.session_state.extracted_text,
                        max_length=max_length
                    )
                    
                    st.subheader("📄 Summary")
                    st.write(summary)
                    
                    # Word count
                    st.caption(f"Summary length: {len(summary.split())} words")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif feature == "📚 Document History":
    st.header("📚 Document History")
    
    if st.session_state.db is None:
        st.session_state.db = load_database()
    
    if st.session_state.db is None:
        st.error("❌ Database not available")
    else:
        try:
            docs = st.session_state.db.get_all_documents(limit=50)
            
            if not docs:
                st.info("No documents found. Upload your first document!")
                if st.button("Go to Upload"):
                    st.session_state.feature = "📤 Upload Document"
                    st.rerun()
            else:
                st.success(f"Total documents: {len(docs)}")
                
                for doc in docs:
                    with st.expander(f"📄 {doc.filename} (ID: {doc.id})", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        col1.write(f"**Type:** {doc.file_type}")
                        col2.write(f"**Size:** {doc.file_size / 1024:.2f} KB")
                        col3.write(f"**Uploaded:** {doc.uploaded_at.strftime('%Y-%m-%d %H:%M')}")
                        
                        col4, col5 = st.columns(2)
                        with col4:
                            if st.button(f"📂 Load Document", key=f"load_{doc.id}"):
                                st.session_state.current_doc_id = doc.id
                                st.session_state.extracted_text = doc.extracted_text
                                st.session_state.current_filename = doc.filename
                                st.session_state.processing_complete = True
                                st.success(f"✅ Loaded document {doc.id}")
                                st.rerun()
                        
                        with col5:
                            if st.button(f"🗑️ Delete", key=f"delete_{doc.id}"):
                                try:
                                    st.session_state.db.delete_document(doc.id)
                                    st.success(f"✅ Deleted document {doc.id}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error deleting document: {e}")
        except Exception as e:
            st.error(f"❌ Error loading document history: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        👨‍💻 Dibuat oleh <b>Paa Meyy</b><br><br>
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
