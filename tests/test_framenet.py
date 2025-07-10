#!/usr/bin/env python3
"""
Test script for the new FrameNet API endpoints.

This script demonstrates all the FrameNet capabilities of the NLP microservice.
"""

from nlpmicroservice.client import NLPClient
from nlpmicroservice.config import settings


def test_framenet_endpoints():
    """Test all FrameNet endpoints."""
    
    print("=" * 70)
    print("         FrameNet API Demonstration")
    print("=" * 70)
    print()
    
    try:
        client = NLPClient(host="localhost", port=settings.server_port)
        client.connect()
        
        try:
            
            # Test 1: Frame Search
            print("🔍 1. FRAME SEARCH")
            print("-" * 30)
            try:
                # Search for frames related to "communication"
                frame_results = client.search_frames("(?i)communication", max_results=5)
                print(f"Found {frame_results['total_count']} frames matching 'communication':")
                
                for frame in frame_results['frames'][:3]:  # Show first 3
                    print(f"  • {frame['name']} (ID: {frame['id']})")
                    print(f"    Definition: {frame['definition'][:100]}...")
                    print(f"    Sample LUs: {', '.join(frame['lexical_units'][:5])}")
                    print()
            except Exception as e:
                print(f"    ❌ Frame search failed: {e}")
                print()
            
            # Test 2: Frame Details
            print("📋 2. FRAME DETAILS")
            print("-" * 30)
            try:
                # Get details for a specific frame
                frame_details = client.get_frame_details(frame_name="Communication")
                
                print(f"Frame: {frame_details['name']} (ID: {frame_details['id']})")
                print(f"Definition: {frame_details['definition'][:150]}...")
                print()
                
                print(f"Lexical Units ({len(frame_details['lexical_units'])}):")
                for lu in frame_details['lexical_units'][:5]:
                    print(f"  • {lu['name']} ({lu['pos']}) - {lu['definition'][:50]}...")
                print()
                
                print(f"Frame Elements ({len(frame_details['frame_elements'])}):")
                for fe in frame_details['frame_elements'][:5]:
                    print(f"  • {fe['name']} ({fe['core_type']}) - {fe['definition'][:50]}...")
                print()
                
            except Exception as e:
                print(f"    ❌ Frame details failed: {e}")
                print()
            
            # Test 3: Lexical Unit Search
            print("📚 3. LEXICAL UNIT SEARCH")
            print("-" * 30)
            try:
                # Search for lexical units related to "speak"
                lu_results = client.search_lexical_units("(?i)speak", max_results=5)
                print(f"Found {lu_results['total_count']} lexical units matching 'speak':")
                
                for lu in lu_results['lexical_units']:
                    print(f"  • {lu['name']} ({lu['pos']}) in frame '{lu['frame_name']}'")
                    print(f"    Definition: {lu['definition'][:80]}...")
                    print()
            except Exception as e:
                print(f"    ❌ Lexical unit search failed: {e}")
                print()
            
            # Test 4: Frame Relations
            print("🔗 4. FRAME RELATIONS")
            print("-" * 30)
            try:
                # Get relations for the Communication frame
                relations_data = client.get_frame_relations(frame_name="Communication")
                
                print(f"Frame relations for 'Communication':")
                print(f"Available relation types: {', '.join(relations_data['relation_types'][:5])}...")
                print()
                
                for rel in relations_data['relations'][:5]:
                    print(f"  • {rel['type']}: {rel['parent_frame']} → {rel['child_frame']}")
                    print(f"    {rel['description']}")
                    print()
            except Exception as e:
                print(f"    ❌ Frame relations failed: {e}")
                print()
            
            # Test 5: Semantic Role Labeling
            print("🎯 5. SEMANTIC ROLE LABELING")
            print("-" * 30)
            try:
                # Test semantic role labeling on sample text
                sample_text = "John told Mary about the meeting yesterday."
                semantic_frames = client.semantic_role_labeling(sample_text, include_frame_elements=True)
                
                print(f"Text: '{sample_text}'")
                print(f"Found {len(semantic_frames)} semantic frames:")
                print()
                
                for frame in semantic_frames:
                    print(f"  Frame: {frame['frame_name']} (ID: {frame['frame_id']})")
                    print(f"  Trigger: '{frame['trigger_word']}' at position {frame['trigger_start']}")
                    
                    if frame['roles']:
                        print(f"  Roles:")
                        for role in frame['roles']:
                            print(f"    • {role['role_name']} ({role['role_type']}): '{role['text']}'")
                    else:
                        print(f"  No roles identified")
                    print()
            except Exception as e:
                print(f"    ❌ Semantic role labeling failed: {e}")
                print()
            
            # Test 6: Advanced Frame Search
            print("🔬 6. ADVANCED FEATURES")
            print("-" * 30)
            try:
                # Search for frames related to emotions
                emotion_frames = client.search_frames("(?i)emotion", max_results=3)
                print(f"Emotion-related frames ({emotion_frames['total_count']} total):")
                
                for frame in emotion_frames['frames']:
                    print(f"  • {frame['name']}")
                    
                    # Get detailed info for this frame
                    details = client.get_frame_details(frame_id=frame['id'])
                    core_fes = [fe['name'] for fe in details['frame_elements'] if fe['core_type'] == 'Core']
                    print(f"    Core elements: {', '.join(core_fes[:3])}")
                    print()
                    
            except Exception as e:
                print(f"    ❌ Advanced features failed: {e}")
                print()
        
        finally:
            client.disconnect()
                
    except Exception as e:
        print(f"❌ Error connecting to NLP service: {e}")
        print("\nMake sure the server is running:")
        print("  poetry run nlp-server")


def test_performance():
    """Test performance of FrameNet endpoints."""
    
    print("=" * 70)
    print("         FrameNet Performance Test")
    print("=" * 70)
    
    import time
    
    operations = [
        ('Frame Search', lambda client: client.search_frames("(?i)communication", 10)),
        ('Frame Details', lambda client: client.get_frame_details(frame_name="Communication")),
        ('LU Search', lambda client: client.search_lexical_units("(?i)speak", max_results=10)),
        ('Frame Relations', lambda client: client.get_frame_relations(frame_name="Communication")),
        ('Semantic Role Labeling', lambda client: client.semantic_role_labeling("John spoke to Mary")),
    ]
    
    try:
        client = NLPClient()
        client.connect()
        
        for op_name, op_func in operations:
            try:
                # Warm up
                op_func(client)
                
                # Benchmark
                start_time = time.time()
                for _ in range(3):  # Run 3 times
                    op_func(client)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 3 * 1000  # ms
                print(f"{op_name:25} {avg_time:8.2f} ms")
            except Exception as e:
                print(f"{op_name:25} ERROR: {e}")
                
        client.disconnect()
                
    except Exception as e:
        print(f"❌ Error during performance test: {e}")


if __name__ == "__main__":
    print("Starting FrameNet API Demo...")
    print()
    
    # Run main demonstration
    test_framenet_endpoints()
    
    print("\n" + "=" * 70)
    input("Press Enter to run performance test...")
    test_performance()
    
    print("\n" + "=" * 70)
    print("Demo completed! 🎉")
    print("FrameNet endpoints are now available in your NLP microservice!")
