"""gRPC server implementation for NLP microservice."""

import asyncio
import grpc
from concurrent import futures
from typing import Iterator
from loguru import logger

from .config import settings
from .nlp_service import NLPProcessor

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


class NLPServicer(nlp_pb2_grpc.NLPServiceServicer):
    """gRPC servicer for NLP operations."""
    
    def __init__(self):
        """Initialize the NLP servicer."""
        self.nlp_processor = NLPProcessor()
        logger.info("NLP servicer initialized")
    
    def Tokenize(self, request, context):
        """Tokenize text into words."""
        try:
            logger.info(f"Tokenizing text: {request.text[:100]}...")
            tokens, count = self.nlp_processor.tokenize(
                request.text, 
                request.language or "english"
            )
            
            response = nlp_pb2.TokenizeResponse()
            response.tokens.extend(tokens)
            response.token_count = count
            
            logger.info(f"Tokenization completed. Token count: {count}")
            return response
        except Exception as e:
            logger.error(f"Error in Tokenize: {e}")
            context.set_details(f"Tokenization failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return nlp_pb2.TokenizeResponse()
    
    def SentimentAnalysis(self, request, context):
        """Perform sentiment analysis on text."""
        try:
            logger.info(f"Analyzing sentiment for text: {request.text[:100]}...")
            sentiment, confidence = self.nlp_processor.analyze_sentiment(request.text)
            
            response = nlp_pb2.SentimentResponse()
            response.sentiment = sentiment
            response.confidence = confidence
            
            logger.info(f"Sentiment analysis completed. Sentiment: {sentiment}, Confidence: {confidence}")
            return response
        except Exception as e:
            logger.error(f"Error in SentimentAnalysis: {e}")
            context.set_details(f"Sentiment analysis failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return nlp_pb2.SentimentResponse()
    
    def NamedEntityRecognition(self, request, context):
        """Extract named entities from text."""
        try:
            logger.info(f"Extracting named entities from text: {request.text[:100]}...")
            entities = self.nlp_processor.extract_named_entities(request.text)
            
            response = nlp_pb2.NERResponse()
            for entity in entities:
                ne = response.entities.add()
                ne.text = entity['text']
                ne.label = entity['label']
                ne.start = entity['start']
                ne.end = entity['end']
            
            logger.info(f"Named entity recognition completed. Found {len(entities)} entities")
            return response
        except Exception as e:
            logger.error(f"Error in NamedEntityRecognition: {e}")
            context.set_details(f"Named entity recognition failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return nlp_pb2.NERResponse()
    
    def POSTagging(self, request, context):
        """Perform part-of-speech tagging."""
        try:
            logger.info(f"Performing POS tagging for text: {request.text[:100]}...")
            pos_tags = self.nlp_processor.pos_tagging(request.text)
            
            response = nlp_pb2.POSResponse()
            for word, tag in pos_tags:
                pos_tag = response.tags.add()
                pos_tag.word = word
                pos_tag.tag = tag
            
            logger.info(f"POS tagging completed. Tagged {len(pos_tags)} words")
            return response
        except Exception as e:
            logger.error(f"Error in POSTagging: {e}")
            context.set_details(f"POS tagging failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return nlp_pb2.POSResponse()
    
    def TextSimilarity(self, request, context):
        """Calculate text similarity."""
        try:
            logger.info(f"Calculating similarity between texts...")
            similarity = self.nlp_processor.calculate_similarity(request.text1, request.text2)
            
            response = nlp_pb2.SimilarityResponse()
            response.similarity = similarity
            
            logger.info(f"Text similarity calculation completed. Similarity: {similarity}")
            return response
        except Exception as e:
            logger.error(f"Error in TextSimilarity: {e}")
            context.set_details(f"Text similarity calculation failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return nlp_pb2.SimilarityResponse()
    
    def KeywordExtraction(self, request, context):
        """Extract keywords from text."""
        try:
            logger.info(f"Extracting keywords from text: {request.text[:100]}...")
            max_keywords = request.max_keywords or 10
            keywords = self.nlp_processor.extract_keywords(request.text, max_keywords)
            
            response = nlp_pb2.KeywordResponse()
            for word, score in keywords:
                keyword = response.keywords.add()
                keyword.word = word
                keyword.score = score
            
            logger.info(f"Keyword extraction completed. Found {len(keywords)} keywords")
            return response
        except Exception as e:
            logger.error(f"Error in KeywordExtraction: {e}")
            context.set_details(f"Keyword extraction failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return nlp_pb2.KeywordResponse()


def serve():
    """Start the gRPC server."""
    logger.info("Starting NLP gRPC server...")
    
    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=settings.max_workers))
    
    # Add servicer to server
    nlp_pb2_grpc.add_NLPServiceServicer_to_server(NLPServicer(), server)
    
    # Configure server address
    listen_addr = f"{settings.server_host}:{settings.server_port}"
    server.add_insecure_port(listen_addr)
    
    # Start server
    server.start()
    logger.info(f"NLP gRPC server started on {listen_addr}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        server.stop(0)


def main():
    """Main entry point for the server."""
    from loguru import logger
    
    # Configure logging
    logger.add(
        "logs/nlp_server.log",
        rotation="1 day",
        retention="30 days",
        level=settings.log_level,
        format=settings.log_format
    )
    
    logger.info("NLP Microservice starting up...")
    logger.info(f"Configuration: {settings.dict()}")
    
    serve()


if __name__ == "__main__":
    main()
