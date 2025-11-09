#ocr_engine.py
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import cv2
import numpy as np
from typing import List, Union
import io

class OCREngine:
    """Optical Character Recognation Engine"""

    def __init__(self, engine: str = "tesseract", lang: str = "eng"):
        self.engine = engine
        self.lang = lang 

    def preprocesse_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Preprocess image for better OCR results"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2GRAY)

        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        denoised = cv2.fastNlMeansDenoising(thresh)
        return denoised

    def extract_text_from_image(self, image_path: str) -> str:
        try:
            image = Image.open(image_path)
            processed = self.preprocesse_image(image)

            if self.engine == "tesseract":
                text = pytesseract.image_to_string(
                    processed,
                    lang=self.lang,
                    config='--psm 6'         
                )
            else:
                from paddleocr import PaddleOCR
                ocr = PaddleOCR(use_angle_cls=True, lang='en')
                result = ocr.ocr(processed, cls=True)
                text = '\n'.join([line[1][0] for line in result[0]])

            return text.strip()
        
        except Exception as e:
            raise Exception(f"Error extracing text from image: {str(e)}")
    
    def extract_tex_from_pdf(self, pdf_path: str) -> str:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                if text.strip():
                    return text.strip()
                
            images = convert_from_path(pdf_path)
            text = ""

            for i, image in enumerate(images):
                print(f"processing page {i+1}/{len(images)}...")
                page_text = self.extract_text_from_image_object(image)
                text += f"\n--- Page {i+1} ---\n{page_text}\n"

            return text.strip()
        
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_text_from_image_object(self, image: Image) -> str:
        """Extract text from PIL Image object"""
        try:
            processed = self.preprocesse_image(image)

            if self.engine == "tesseract":
                text = pytesseract.image_to_string(
                    processed,
                    lang=self.lang,
                    config='--psm 6'
                )
            else:
                from paddleocr import PaddleOCR
                ocr = PaddleOCR(use_angle_cls=True, lang='en')
                result = ocr.ocr(np.array(processed), cls=True)
                text = '\n'.join([line[1][0] for line in result[0]])

            return text.strip()

        except Exception as e:
            raise Exception(f"Error extracting text: {str(e)}")

    def extract_text(self, file_path: str) -> str:
        """Main method to extract text from any supported file"""
        file_extension = file_path.lower().split('.')[-1]

        if file_extension == 'pdf':
            return self.extract_tex_from_pdf(file_path)
        elif file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.head()
        else:
            raise ValueError(f"Unsuported file type: {file_extension}")