"""Client for testing the NLP gRPC service."""

import grpc
from typing import List, Dict, Any
from loguru import logger

from .config import settings

# Import generated gRPC modules (will be generated from proto files)
try:
    from . import nlp_pb2
    from . import nlp_pb2_grpc
except ImportError:
    logger.warning("Generated gRPC modules not found. Run 'make generate-proto' to generate them.")
    # Create placeholder classes for development
    class nlp_pb2:
        pass
    class nlp_pb2_grpc:
        pass


class NLPClient:
    """Client for the NLP gRPC service."""
    
    def __init__(self, host: str = "localhost", port: int = 8161):
        """Initialize the NLP client."""
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None
    
    def connect(self):
        """Connect to the gRPC server."""
        self.channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        self.stub = nlp_pb2_grpc.NLPServiceStub(self.channel)
        logger.info(f"Connected to NLP service at {self.host}:{self.port}")
    
    def disconnect(self):
        """Disconnect from the gRPC server."""
        if self.channel:
            self.channel.close()
            logger.info("Disconnected from NLP service")
    
    def tokenize(self, text: str, language: str = "english") -> Dict[str, Any]:
        """Tokenize text."""
        if not self.stub:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        request = nlp_pb2.TokenizeRequest()
        request.text = text
        request.language = language
        
        response = self.stub.Tokenize(request)
        return {
            "tokens": list(response.tokens),
            "token_count": response.token_count
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        if not self.stub:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        request = nlp_pb2.SentimentRequest()
        request.text = text
        
        response = self.stub.SentimentAnalysis(request)
        return {
            "sentiment": response.sentiment,
            "confidence": response.confidence
        }
    
    def extract_named_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        if not self.stub:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        request = nlp_pb2.NERRequest()
        request.text = text
        
        response = self.stub.NamedEntityRecognition(request)
        return [
            {
                "text": entity.text,
                "label": entity.label,
                "start": entity.start,
                "end": entity.end
            }
            for entity in response.entities
        ]
    
    def pos_tagging(self, text: str) -> List[Dict[str, str]]:
        """Perform part-of-speech tagging."""
        if not self.stub:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        request = nlp_pb2.POSRequest()
        request.text = text
        
        response = self.stub.POSTagging(request)
        return [
            {
                "word": tag.word,
                "tag": tag.tag
            }
            for tag in response.tags
        ]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity."""
        if not self.stub:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        request = nlp_pb2.SimilarityRequest()
        request.text1 = text1
        request.text2 = text2
        
        response = self.stub.TextSimilarity(request)
        return response.similarity
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[Dict[str, Any]]:
        """Extract keywords from text."""
        if not self.stub:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        request = nlp_pb2.KeywordRequest()
        request.text = text
        request.max_keywords = max_keywords
        
        response = self.stub.KeywordExtraction(request)
        return [
            {
                "word": keyword.word,
                "score": keyword.score
            }
            for keyword in response.keywords
        ]
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def main():
    """Example usage of the NLP client."""
    # Example text for testing
    test_text = """
    Natural Language Processing (NLP) is a fascinating field that combines 
    computer science, artificial intelligence, and linguistics. It enables 
    computers to understand, interpret, and generate human language in a 
    valuable way. Companies like Google, Microsoft, and Apple use NLP 
    technologies in their products.
    """
    
    with NLPClient() as client:
        print("=== NLP Service Client Example ===\n")
        
        # Tokenization
        print("1. Tokenization:")
        tokens_result = client.tokenize(test_text)
        print(f"   Tokens: {tokens_result['tokens'][:10]}...")
        print(f"   Token count: {tokens_result['token_count']}\n")
        
        # Sentiment Analysis
        print("2. Sentiment Analysis:")
        sentiment_result = client.analyze_sentiment(test_text)
        print(f"   Sentiment: {sentiment_result['sentiment']}")
        print(f"   Confidence: {sentiment_result['confidence']:.3f}\n")
        
        # Named Entity Recognition
        print("3. Named Entity Recognition:")
        entities = client.extract_named_entities(test_text)
        for entity in entities:
            print(f"   {entity['text']} ({entity['label']})")
        print()
        
        # POS Tagging
        print("4. POS Tagging (first 10 words):")
        pos_tags = client.pos_tagging(test_text)
        for tag in pos_tags[:10]:
            print(f"   {tag['word']}: {tag['tag']}")
        print()
        
        # Text Similarity
        print("5. Text Similarity:")
        text1 = "Natural Language Processing is amazing"
        text2 = "NLP is a fascinating field"
        similarity = client.calculate_similarity(text1, text2)
        print(f"   Similarity between texts: {similarity:.3f}\n")
        
        # Keyword Extraction
        print("6. Keyword Extraction:")
        keywords = client.extract_keywords(test_text, max_keywords=5)
        for keyword in keywords:
            print(f"   {keyword['word']}: {keyword['score']:.3f}")


if __name__ == "__main__":
    main()
