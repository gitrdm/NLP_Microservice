"""NLP service implementation using NLTK."""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from typing import List, Tuple, Dict, Any
import math
import re
from loguru import logger

from .config import settings


class NLPProcessor:
    """Natural Language Processing service using NLTK."""
    
    def __init__(self):
        """Initialize the NLP processor."""
        self._download_nltk_data()
        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def _download_nltk_data(self) -> None:
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('chunkers/maxent_ne_chunker')
            nltk.data.find('corpora/words')
            nltk.data.find('vader_lexicon')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            logger.info("NLTK data already available")
        except LookupError:
            logger.info("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            logger.info("NLTK data downloaded successfully")
    
    def tokenize(self, text: str, language: str = "english") -> Tuple[List[str], int]:
        """Tokenize text into words."""
        try:
            tokens = word_tokenize(text, language=language)
            # Remove punctuation and convert to lowercase
            tokens = [token.lower() for token in tokens if token.isalnum()]
            return tokens, len(tokens)
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment of text."""
        try:
            scores = self.sia.polarity_scores(text)
            compound_score = scores['compound']
            
            if compound_score >= 0.05:
                sentiment = "positive"
            elif compound_score <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            confidence = abs(compound_score)
            return sentiment, confidence
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            raise
    
    def extract_named_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)
            
            entities = []
            current_entity = []
            current_label = None
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    # This is a named entity
                    if current_label != chunk.label():
                        # New entity type
                        if current_entity:
                            entity_text = ' '.join([word for word, _ in current_entity])
                            entities.append({
                                'text': entity_text,
                                'label': current_label,
                                'start': 0,  # Simplified for now
                                'end': 0     # Simplified for now
                            })
                        current_entity = list(chunk)
                        current_label = chunk.label()
                    else:
                        # Continue current entity
                        current_entity.extend(list(chunk))
                else:
                    # Not a named entity
                    if current_entity:
                        entity_text = ' '.join([word for word, _ in current_entity])
                        entities.append({
                            'text': entity_text,
                            'label': current_label,
                            'start': 0,  # Simplified for now
                            'end': 0     # Simplified for now
                        })
                        current_entity = []
                        current_label = None
            
            # Handle last entity
            if current_entity:
                entity_text = ' '.join([word for word, _ in current_entity])
                entities.append({
                    'text': entity_text,
                    'label': current_label,
                    'start': 0,  # Simplified for now
                    'end': 0     # Simplified for now
                })
            
            return entities
        except Exception as e:
            logger.error(f"Error extracting named entities: {e}")
            raise
    
    def pos_tagging(self, text: str) -> List[Tuple[str, str]]:
        """Perform part-of-speech tagging."""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            return pos_tags
        except Exception as e:
            logger.error(f"Error performing POS tagging: {e}")
            raise
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using cosine similarity."""
        try:
            # Tokenize and process texts
            tokens1 = self.tokenize(text1)[0]
            tokens2 = self.tokenize(text2)[0]
            
            # Remove stop words and lemmatize
            processed1 = [self.lemmatizer.lemmatize(token) for token in tokens1 
                         if token not in self.stop_words]
            processed2 = [self.lemmatizer.lemmatize(token) for token in tokens2 
                         if token not in self.stop_words]
            
            # Create frequency vectors
            all_words = set(processed1 + processed2)
            vector1 = [processed1.count(word) for word in all_words]
            vector2 = [processed2.count(word) for word in all_words]
            
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(vector1, vector2))
            magnitude1 = math.sqrt(sum(a * a for a in vector1))
            magnitude2 = math.sqrt(sum(b * b for b in vector2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            similarity = dot_product / (magnitude1 * magnitude2)
            return similarity
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            raise
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords from text using TF-IDF approach."""
        try:
            # Tokenize and clean text
            tokens = self.tokenize(text)[0]
            
            # Remove stop words and lemmatize
            processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                              if token not in self.stop_words and len(token) > 2]
            
            # Calculate term frequency
            term_freq = Counter(processed_tokens)
            total_terms = len(processed_tokens)
            
            # Simple TF score (normalized frequency)
            keywords = []
            for term, freq in term_freq.most_common(max_keywords):
                tf_score = freq / total_terms
                keywords.append((term, tf_score))
            
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            raise
