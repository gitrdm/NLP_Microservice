#!/usr/bin/env python3
"""
Test script to verify server and client integration.
This script will start a server in the background and run a quick client test.
"""

import sys
import time
import subprocess
import signal
from pathlib import Path
from multiprocessing import Process

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from nlpmicroservice.client import NLPClient
from loguru import logger


def start_server():
    """Start the server in a subprocess."""
    try:
        # Start server process
        server_process = subprocess.Popen(
            [sys.executable, "run_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent.parent
        )
        
        # Give server time to start
        time.sleep(3)
        
        return server_process
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return None


def test_client_integration():
    """Test client integration with the server."""
    logger.info("Starting integration test...")
    
    # Start server
    server_process = start_server()
    if not server_process:
        logger.error("Failed to start server")
        return False
    
    try:
        # Give server time to initialize
        time.sleep(2)
        
        # Test client connection
        client = NLPClient(host="localhost", port=8161)
        client.connect()
        
        # Test WordNet functionality
        logger.info("Testing WordNet synset lookup...")
        result = client.lookup_synsets("dog")
        assert "synsets" in result
        assert "word" in result
        assert len(result["synsets"]) > 0
        logger.info(f"✓ Found {len(result['synsets'])} synsets for 'dog'")
        
        logger.info("Testing WordNet synset details...")
        details = client.get_synset_details("dog.n.01")
        assert "synset" in details
        assert "lemmas" in details
        assert "hypernyms" in details
        logger.info(f"✓ Got details for {details['synset']['id']}")
        
        logger.info("Testing WordNet similarity...")
        similarity = client.calculate_synset_similarity("dog.n.01", "cat.n.01")
        assert "similarity_score" in similarity
        assert "synset1_id" in similarity
        assert "synset2_id" in similarity
        logger.info(f"✓ Calculated similarity: {similarity['similarity_score']:.4f}")
        
        client.disconnect()
        
        logger.info("🎉 All integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False
        
    finally:
        # Clean up server
        if server_process:
            try:
                server_process.terminate()
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
            logger.info("Server process terminated")


if __name__ == "__main__":
    success = test_client_integration()
    sys.exit(0 if success else 1)
