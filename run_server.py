#!/usr/bin/env python3
"""
Server startup script for the NLP microservice.
"""

import sys
import time
import signal
import grpc
from concurrent import futures
from pathlib import Path
from loguru import logger

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from nlpmicroservice.server import NLPServicer
from nlpmicroservice import nlp_pb2_grpc
from nlpmicroservice.config import settings


def serve():
    """Start the gRPC server."""
    # Create server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add servicer
    nlp_pb2_grpc.add_NLPServiceServicer_to_server(NLPServicer(), server)
    
    # Bind to port
    port = getattr(settings, 'GRPC_PORT', 8161)
    server.add_insecure_port(f'[::]:{port}')
    
    # Start server
    server.start()
    logger.info(f"🚀 NLP microservice server started on port {port}")
    logger.info("Available endpoints:")
    logger.info("  - TokenizeRequest")
    logger.info("  - SentimentRequest")
    logger.info("  - NERRequest")
    logger.info("  - POSRequest")
    logger.info("  - SimilarityRequest")
    logger.info("  - KeywordRequest")
    logger.info("  - SummarizationRequest")
    logger.info("  - FrameSearchRequest")
    logger.info("  - FrameDetailsRequest")
    logger.info("  - LexicalUnitSearchRequest")
    logger.info("  - FrameRelationsRequest")
    logger.info("  - SemanticRoleLabelingRequest")
    logger.info("  - SynsetsLookupRequest")
    logger.info("  - SynsetDetailsRequest")
    logger.info("  - SynsetSimilarityRequest")
    logger.info("  - SynsetRelationsRequest")
    logger.info("  - LemmaSearchRequest")
    logger.info("  - SynonymSearchRequest")
    logger.info("Server is ready to accept requests...")
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, stopping server...")
        server.stop(0)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Keep server running
        while True:
            time.sleep(86400)  # Sleep for 24 hours
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)


if __name__ == "__main__":
    serve()
