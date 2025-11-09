#nlp_engine.py
from transformers import pipeline
import spacy
from typing import List, Dict
import re

class NLPEngine:
    """Natural Language Processing Engine"""

    def __init__(self):
        print("Loading NLP models...")

        # Question answering pipeline
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2"
        )

        # Summarization pipeline
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )

        # spaCy NLP model
        self.nlp = spacy.load("en_core_web_sm")

        # Sentiment analysis pipeline (correct model name)
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        print("NLP models loaded successfully!")

    def answer_question(self, question: str, context: str, max_length: int = 512) -> Dict:
        """Answer questions based on document context"""
        try:
            tokenizer = getattr(self.qa_pipeline, "tokenizer", None)
            if tokenizer:
                tokens = tokenizer.encode(context, truncation=True, max_length=max_length)
                context = tokenizer.decode(tokens, skip_special_tokens=True)

            result = self.qa_pipeline(
                question=question,
                context=context
            )

            # result is usually a dict for question-answering pipeline
            return {
                "answer": result.get("answer", ""),
                "confidence": round(result.get("score", 0.0), 4),
                "start": result.get("start", None),
                "end": result.get("end", None)
            }
        except Exception as e:
            return {
                "answer": "Unable to find answer",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """Generate summary of the text"""
        try:
            max_chunk_length = 1024
            chunks = self._split_text(text, max_chunk_length)

            summaries = []
            for chunk in chunks:
                if len(chunk.split()) < 50:
                    continue

                summary = self.summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                if isinstance(summary, list) and summary:
                    summaries.append(summary[0].get("summary_text", ""))
                elif isinstance(summary, dict):
                    summaries.append(summary.get("summary_text", ""))

            final_summary = " ".join([s for s in summaries if s])
            return final_summary
    
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities (people, organizations, locations, dates, etc.)"""
        try:
            doc = self.nlp(text)
            entities: Dict[str, List[str]] = {
                "PERSON": [],
                "ORG": [],
                "GPE": [],
                "DATE": [],
                "MONEY": [],
                "LOCATION": [],
                "OTHER": []
            }

            for ent in doc.ents:
                label = ent.label_
                ent_text = ent.text.strip()

                if label in entities:
                    if ent_text not in entities[label]:
                        entities[label].append(ent_text)
                else:
                    if ent_text not in entities["OTHER"]:
                        entities["OTHER"].append(ent_text)

            entities = {k: v for k, v in entities.items() if v}
            return entities
        except Exception as e:
            return {"error": str(e)}
           
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of the text"""
        try:
            max_length = 512
            chunks = self._split_text(text, max_length)

            sentiments = []
            for chunk in chunks[:5]:
                if not chunk.strip():
                    continue
                result = self.sentiment_analyzer(chunk)
                if isinstance(result, list) and result:
                    sentiments.append(result[0])
                elif isinstance(result, dict):
                    sentiments.append(result)

            if not sentiments:
                return {"error": "No text to analyze"}

            positive_count = sum(1 for s in sentiments if s.get("label") == "POSITIVE")
            negative_count = len(sentiments) - positive_count
            avg_score = sum(s.get("score", 0.0) for s in sentiments) / len(sentiments)

            overall_sentiment = "POSITIVE" if positive_count > negative_count else "NEGATIVE"

            return {
                "overall_sentiment": overall_sentiment,
                "confidence": round(avg_score, 4),
                "positive_chunks": positive_count,
                "negative_chunks": negative_count,
                "details": sentiments
            }
        
        except Exception as e:
            return {"error": str(e)}
        
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key phrases from text"""
        try:
            doc = self.nlp(text)

            keywords = set()

            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:
                    keywords.add(chunk.text.lower().strip())
                     
            keywords_list = list(keywords)[:top_n]

            return keywords_list

        except Exception:
            return []

    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks of maximum length (approximate by characters)"""
        if not text:
            return []

        words = text.split()
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for word in words:
            # +1 for the space
            word_len = len(word) + 1
            if current_length + word_len > max_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word) + 1
            else:
                current_chunk.append(word)
                current_length += word_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def search_in_text(self, text: str, query: str) -> List[Dict]:
        """Search for query in text and return relevant passages"""
        try:
            query_lower = query.lower()
            sentences = re.split(r'([.!?])', text)

            # Reconstruct sentences with their punctuation
            full_sentences = []
            for i in range(0, len(sentences) - 1, 2):
                sent = (sentences[i] + sentences[i+1]).strip()
                if sent:
                    full_sentences.append(sent)
            # handle trailing fragment
            if len(sentences) % 2 == 1 and sentences[-1].strip():
                full_sentences.append(sentences[-1].strip())
            
            results = []
            for i, sentence in enumerate(full_sentences):
                if query_lower in sentence.lower():
                    context_start = max(0, i-1)
                    context_end = min(len(full_sentences), i+2)
                    context = " ".join(full_sentences[context_start:context_end])

                    results.append({
                        "sentence": sentence.strip(),
                        "context": context.strip(),
                        "position": i
                    })
            return results
        
        except Exception as e:
            return {"error": str(e)}