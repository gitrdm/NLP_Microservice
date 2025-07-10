#!/usr/bin/env python3
"""
Simple server test script to diagnose server startup issues.
"""

import sys
import traceback
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    print("1. Testing imports...")
    
    import grpc
    from concurrent import futures
    print("✓ gRPC imports successful")
    
    from nlpmicroservice.server import NLPServicer
    print("✓ Server class import successful")
    
    from nlpmicroservice import nlp_pb2_grpc
    print("✓ gRPC modules import successful")
    
    print("\n2. Testing server creation...")
    
    # Create server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    print("✓ gRPC server created")
    
    # Add servicer
    nlp_pb2_grpc.add_NLPServiceServicer_to_server(NLPServicer(), server)
    print("✓ NLP servicer added")
    
    print("\n3. Testing port binding...")
    
    # Try to bind to port
    port = server.add_insecure_port('[::]:0')  # Let system choose port
    print(f"✓ Server bound to port {port}")
    
    print("\n4. Testing server start...")
    
    # Start server
    server.start()
    print("✓ Server started successfully")
    
    # Stop server
    server.stop(0)
    print("✓ Server stopped successfully")
    
    print("\n🎉 All server tests passed! Server should work correctly.")
    
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    print(f"Error type: {type(e).__name__}")
    print("\nFull traceback:")
    traceback.print_exc()
    
    print("\n🔍 Debugging suggestions:")
    print("1. Check if all dependencies are installed: pip install -r requirements.txt")
    print("2. Verify gRPC installation: pip install grpcio grpcio-tools")
    print("3. Check if proto files are generated: make generate-proto")
    print("4. Ensure NLTK data is available")
    print("5. Check for port conflicts")
    
    sys.exit(1)
