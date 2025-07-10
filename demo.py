#!/usr/bin/env python3
"""
Example demonstration of the NLP Microservice.

This script shows how to use the NLP microservice for various text processing tasks.
"""

import asyncio
import time
from nlpmicroservice.client import NLPClient
from nlpmicroservice.config import settings


def demo_nlp_service():
    """Demonstrate the NLP microservice capabilities."""
    
    # Sample texts for demonstration
    sample_texts = [
        "I absolutely love this new product! It's amazing and works perfectly.",
        "This service is terrible and completely broken. Very disappointed.",
        "The weather is nice today. I might go for a walk in the park.",
        "Apple Inc. is a technology company based in Cupertino, California. Tim Cook is the CEO.",
        "Machine learning and artificial intelligence are transforming healthcare."
    ]
    
    print("=" * 60)
    print("         NLP Microservice Demonstration")
    print("=" * 60)
    print()
    
    try:
        with NLPClient(host="localhost", port=settings.server_port) as client:
            
            for i, text in enumerate(sample_texts, 1):
                print(f"Text {i}: {text}")
                print("-" * 50)
                
                # Tokenization
                tokens_result = client.tokenize(text)
                print(f"🔤 Tokens ({tokens_result['token_count']}): {tokens_result['tokens'][:8]}...")
                
                # Sentiment Analysis
                sentiment_result = client.analyze_sentiment(text)
                sentiment_emoji = {
                    'positive': '😊',
                    'negative': '😞',
                    'neutral': '😐'
                }.get(sentiment_result['sentiment'], '❓')
                print(f"{sentiment_emoji} Sentiment: {sentiment_result['sentiment'].upper()} "
                      f"(confidence: {sentiment_result['confidence']:.3f})")
                
                # Named Entity Recognition
                entities = client.extract_named_entities(text)
                if entities:
                    entity_strs = [f"{e['text']} ({e['label']})" for e in entities]
                    print(f"🏷️  Named Entities: {', '.join(entity_strs)}")
                else:
                    print("🏷️  Named Entities: None found")
                
                # Keywords
                keywords = client.extract_keywords(text, max_keywords=3)
                if keywords:
                    keyword_strs = [f"{kw['word']} ({kw['score']:.3f})" for kw in keywords]
                    print(f"🔑 Keywords: {', '.join(keyword_strs)}")
                else:
                    print("🔑 Keywords: None found")
                
                print()
            
            # Text similarity demonstration
            print("=" * 60)
            print("         Text Similarity Comparison")
            print("=" * 60)
            
            similarity_pairs = [
                ("I love programming", "Programming is my passion"),
                ("The cat sat on the mat", "A feline was on the rug"),
                ("Weather is nice today", "The stock market is rising"),
            ]
            
            for text1, text2 in similarity_pairs:
                similarity = client.calculate_similarity(text1, text2)
                print(f"Text 1: {text1}")
                print(f"Text 2: {text2}")
                print(f"Similarity: {similarity:.3f} {'🔥' if similarity > 0.5 else '❄️'}")
                print()
                
    except Exception as e:
        print(f"❌ Error connecting to NLP service: {e}")
        print("\nMake sure the server is running:")
        print("  poetry run nlp-server")
        print("  or")
        print("  python -m nlpmicroservice.server")


def benchmark_performance():
    """Benchmark the NLP service performance."""
    
    print("=" * 60)
    print("         Performance Benchmark")
    print("=" * 60)
    
    test_text = "Natural language processing is a fascinating field that combines computer science and linguistics."
    
    operations = [
        ('Tokenization', lambda client: client.tokenize(test_text)),
        ('Sentiment Analysis', lambda client: client.analyze_sentiment(test_text)),
        ('Named Entity Recognition', lambda client: client.extract_named_entities(test_text)),
        ('POS Tagging', lambda client: client.pos_tagging(test_text)),
        ('Keyword Extraction', lambda client: client.extract_keywords(test_text)),
    ]
    
    try:
        with NLPClient() as client:
            for op_name, op_func in operations:
                # Warm up
                op_func(client)
                
                # Benchmark
                start_time = time.time()
                for _ in range(10):
                    op_func(client)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10 * 1000  # ms
                print(f"{op_name:25} {avg_time:8.2f} ms")
                
    except Exception as e:
        print(f"❌ Error during benchmark: {e}")


if __name__ == "__main__":
    print("Starting NLP Microservice Demo...")
    print()
    
    # Run demonstration
    demo_nlp_service()
    
    print("\n" + "=" * 60)
    input("Press Enter to run performance benchmark...")
    benchmark_performance()
    
    print("\n" + "=" * 60)
    print("Demo completed! 🎉")
    print("Check the logs directory for detailed server logs.")
