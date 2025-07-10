"""Test suite for the NLP microservice."""

import pytest
import grpc
from unittest.mock import Mock, patch, MagicMock
from concurrent import futures
import time

from nlpmicroservice.server import NLPServicer, serve
from nlpmicroservice.nlp_service import NLPProcessor
from nlpmicroservice.client import NLPClient
from nlpmicroservice.config import settings


class TestNLPProcessor:
    """Test cases for NLPProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = NLPProcessor()
    
    def test_tokenize(self):
        """Test text tokenization."""
        text = "Hello world! This is a test."
        tokens, count = self.processor.tokenize(text)
        
        assert isinstance(tokens, list)
        assert isinstance(count, int)
        assert count > 0
        assert "hello" in tokens
        assert "world" in tokens
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        positive_text = "I love this amazing product!"
        negative_text = "This is terrible and awful."
        neutral_text = "This is a table."
        
        pos_sentiment, pos_confidence = self.processor.analyze_sentiment(positive_text)
        neg_sentiment, neg_confidence = self.processor.analyze_sentiment(negative_text)
        neu_sentiment, neu_confidence = self.processor.analyze_sentiment(neutral_text)
        
        assert pos_sentiment == "positive"
        assert neg_sentiment == "negative"
        assert neu_sentiment == "neutral"
        assert 0 <= pos_confidence <= 1
        assert 0 <= neg_confidence <= 1
        assert 0 <= neu_confidence <= 1
    
    def test_pos_tagging(self):
        """Test part-of-speech tagging."""
        text = "The quick brown fox jumps."
        pos_tags = self.processor.pos_tagging(text)
        
        assert isinstance(pos_tags, list)
        assert len(pos_tags) > 0
        assert all(isinstance(tag, tuple) and len(tag) == 2 for tag in pos_tags)
    
    def test_calculate_similarity(self):
        """Test text similarity calculation."""
        text1 = "The cat sat on the mat."
        text2 = "A cat was sitting on a mat."
        text3 = "The dog ran in the park."
        
        similarity_high = self.processor.calculate_similarity(text1, text2)
        similarity_low = self.processor.calculate_similarity(text1, text3)
        
        assert 0 <= similarity_high <= 1
        assert 0 <= similarity_low <= 1
        assert similarity_high > similarity_low
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        text = "Natural language processing is a fascinating field of artificial intelligence."
        keywords = self.processor.extract_keywords(text, max_keywords=5)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        assert all(isinstance(kw, tuple) and len(kw) == 2 for kw in keywords)
        assert all(isinstance(kw[0], str) and isinstance(kw[1], float) for kw in keywords)


class TestNLPServicer:
    """Test cases for NLPServicer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.servicer = NLPServicer()
        self.mock_context = Mock()
    
    @patch('nlpmicroservice.server.nlp_pb2')
    def test_tokenize(self, mock_pb2):
        """Test tokenize RPC method."""
        # Mock request
        mock_request = Mock()
        mock_request.text = "Hello world"
        mock_request.language = "english"
        
        # Mock response
        mock_response = Mock()
        mock_response.tokens = []
        mock_response.token_count = 0
        mock_pb2.TokenizeResponse.return_value = mock_response
        
        # Call method
        response = self.servicer.Tokenize(mock_request, self.mock_context)
        
        # Assertions
        assert response == mock_response
        mock_pb2.TokenizeResponse.assert_called_once()
    
    @patch('nlpmicroservice.server.nlp_pb2')
    def test_sentiment_analysis(self, mock_pb2):
        """Test sentiment analysis RPC method."""
        # Mock request
        mock_request = Mock()
        mock_request.text = "I love this product!"
        
        # Mock response
        mock_response = Mock()
        mock_response.sentiment = "positive"
        mock_response.confidence = 0.8
        mock_pb2.SentimentResponse.return_value = mock_response
        
        # Call method
        response = self.servicer.SentimentAnalysis(mock_request, self.mock_context)
        
        # Assertions
        assert response == mock_response
        mock_pb2.SentimentResponse.assert_called_once()


class TestNLPClient:
    """Test cases for NLPClient."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = NLPClient()
    
    @patch('nlpmicroservice.client.grpc.insecure_channel')
    @patch('nlpmicroservice.client.nlp_pb2_grpc.NLPServiceStub')
    def test_connect(self, mock_stub, mock_channel):
        """Test client connection."""
        mock_channel_instance = Mock()
        mock_channel.return_value = mock_channel_instance
        mock_stub_instance = Mock()
        mock_stub.return_value = mock_stub_instance
        
        self.client.connect()
        
        mock_channel.assert_called_once_with("localhost:8161")
        mock_stub.assert_called_once_with(mock_channel_instance)
        assert self.client.channel == mock_channel_instance
        assert self.client.stub == mock_stub_instance
    
    def test_disconnect(self):
        """Test client disconnection."""
        mock_channel = Mock()
        self.client.channel = mock_channel
        
        self.client.disconnect()
        
        mock_channel.close.assert_called_once()


@pytest.fixture
def grpc_server():
    """Fixture to create a test gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    from nlpmicroservice.server import nlp_pb2_grpc
    
    servicer = NLPServicer()
    nlp_pb2_grpc.add_NLPServiceServicer_to_server(servicer, server)
    
    port = server.add_insecure_port('[::]:0')
    server.start()
    
    yield server, port
    
    server.stop(None)


class TestIntegration:
    """Integration tests for the NLP microservice."""
    
    @pytest.mark.integration
    def test_full_service_flow(self, grpc_server):
        """Test the complete service flow."""
        server, port = grpc_server
        
        # Create client and connect
        client = NLPClient(host="localhost", port=port)
        client.connect()
        
        try:
            # Test tokenization
            result = client.tokenize("Hello world!")
            assert "tokens" in result
            assert "token_count" in result
            assert result["token_count"] > 0
            
            # Test sentiment analysis
            result = client.analyze_sentiment("I love this!")
            assert "sentiment" in result
            assert "confidence" in result
            assert result["sentiment"] in ["positive", "negative", "neutral"]
            
        finally:
            client.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
