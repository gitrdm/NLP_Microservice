#!/usr/bin/env python3
"""Test script for the new TextSummarization endpoint."""

from nlpmicroservice.client import NLPClient
from nlpmicroservice.config import settings

def test_text_summarization():
    """Test the new text summarization endpoint."""
    
    # Sample long text for summarization
    sample_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
    concerned with the interactions between computers and human language, in particular how to program computers 
    to process and analyze large amounts of natural language data. The goal is a computer capable of understanding 
    the contents of documents, including the contextual nuances of the language within them. The technology can 
    then accurately extract information and insights contained in the documents as well as categorize and organize 
    the documents themselves. Challenges in natural language processing frequently involve speech recognition, 
    natural language understanding, and natural language generation. Natural language processing has its roots 
    in the 1950s. Already in 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" 
    which proposed what is now called the Turing test as a criterion of intelligence. The Georgetown experiment 
    in 1954 involved fully automatic translation of more than sixty Russian sentences into English. The authors 
    claimed that within three or five years, machine translation would be a solved problem. However, real progress 
    was much slower, and after the ALPAC report in 1966, which found that ten-year-long research had failed to 
    fulfill the expectations, funding for machine translation was dramatically reduced. Little further research 
    in machine translation was conducted until the late 1980s when the first statistical machine translation 
    systems were developed.
    """
    
    print("Testing Text Summarization Endpoint")
    print("=" * 50)
    print(f"Original text length: {len(sample_text)} characters")
    print(f"Original text: {sample_text[:200]}...")
    print()
    
    try:
        with NLPClient(host="localhost", port=settings.server_port) as client:
            
            # Test with different max_sentences values
            for max_sentences in [2, 3, 5]:
                print(f"Testing with max_sentences={max_sentences}")
                print("-" * 30)
                
                result = client.summarize_text(sample_text, max_sentences=max_sentences)
                
                print(f"Summary: {result['summary']}")
                print(f"Original sentences: {result['original_sentence_count']}")
                print(f"Summary sentences: {result['summary_sentence_count']}")
                print()
                
    except Exception as e:
        print(f"❌ Error testing summarization: {e}")
        print("\nMake sure the server is running:")
        print("  poetry run nlp-server")

if __name__ == "__main__":
    test_text_summarization()
