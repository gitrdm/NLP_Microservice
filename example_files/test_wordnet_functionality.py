#!/usr/bin/env python3
"""
Test script to verify that the WordNet demo script works correctly.
This will test the core functionality without requiring a server.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from nlpmicroservice.nlp_service import NLPProcessor
from loguru import logger


def test_wordnet_functionality():
    """Test that WordNet functionality works as expected."""
    logger.info("Testing WordNet functionality...")
    
    processor = NLPProcessor()
    
    # Test synset lookup
    logger.info("Testing synset lookup...")
    synsets = processor.lookup_synsets("dog")
    assert len(synsets) > 0, "Should find synsets for 'dog'"
    assert synsets[0]["name"] == "dog", "First synset should be 'dog'"
    logger.info(f"✓ Found {len(synsets)} synsets for 'dog'")
    
    # Test synset details
    logger.info("Testing synset details...")
    details = processor.get_synset_details("dog.n.01")
    assert details["synset"]["id"] == "dog.n.01", "Should get details for dog.n.01"
    assert len(details["lemmas"]) > 0, "Should have lemmas"
    assert len(details["hypernyms"]) > 0, "Should have hypernyms"
    logger.info(f"✓ Got details for {details['synset']['id']}")
    
    # Test similarity calculation
    logger.info("Testing similarity calculation...")
    similarity = processor.calculate_synset_similarity("dog.n.01", "cat.n.01")
    assert similarity["similarity_score"] > 0, "Should have positive similarity"
    assert similarity["synset1_id"] == "dog.n.01", "Should maintain synset1_id"
    assert similarity["synset2_id"] == "cat.n.01", "Should maintain synset2_id"
    logger.info(f"✓ Calculated similarity: {similarity['similarity_score']:.4f}")
    
    # Test relations
    logger.info("Testing relations...")
    relations = processor.get_synset_relations("dog.n.01")
    assert relations["synset_id"] == "dog.n.01", "Should maintain synset_id"
    assert len(relations["related_synsets"]) > 0, "Should have related synsets"
    logger.info(f"✓ Found {len(relations['related_synsets'])} related synsets")
    
    # Test lemma search
    logger.info("Testing lemma search...")
    lemmas = processor.search_lemmas("dog")
    assert len(lemmas["lemmas"]) > 0, "Should find lemmas"
    assert lemmas["original_word"] == "dog", "Should maintain original word"
    logger.info(f"✓ Found {len(lemmas['lemmas'])} lemmas")
    
    # Test synonym search
    logger.info("Testing synonym search...")
    synonyms = processor.search_synonyms("dog")
    assert len(synonyms["synonym_groups"]) > 0, "Should find synonym groups"
    assert synonyms["word"] == "dog", "Should maintain search word"
    logger.info(f"✓ Found {len(synonyms['synonym_groups'])} synonym groups")
    
    logger.info("All WordNet functionality tests passed! ✓")


if __name__ == "__main__":
    test_wordnet_functionality()
