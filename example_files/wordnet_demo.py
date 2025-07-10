#!/usr/bin/env python3
"""
Demo script for WordNet/Synset endpoints in the NLP microservice.

This script demonstrates how to use the various WordNet endpoints
including synset lookup, similarity calculation, and relation exploration.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from nlpmicroservice.client import NLPClient
from nlpmicroservice.nlp_service import NLPProcessor
from loguru import logger


def print_separator(title: str):
    """Print a decorative separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_json(data: Dict[str, Any], indent: int = 2):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=indent, ensure_ascii=False))


def demo_synset_lookup(client: NLPClient):
    """Demo synset lookup functionality."""
    print_separator("SYNSET LOOKUP DEMO")
    
    words = ["dog", "cat", "run", "beautiful", "quickly"]
    
    for word in words:
        try:
            print(f"\nLooking up synsets for '{word}':")
            result = client.lookup_synsets(word)
            
            print(f"  Found {len(result['synsets'])} synset(s)")
            
            for i, synset in enumerate(result['synsets'][:3]):  # Show first 3
                print(f"  {i+1}. {synset['name']} ({synset['pos']})")
                print(f"     Definition: {synset['definition']}")
                print(f"     Lemmas: {', '.join(synset['lemma_names'])}")
                if synset['examples']:
                    print(f"     Example: {synset['examples'][0]}")
                
        except Exception as e:
            print(f"  Error looking up '{word}': {e}")


def demo_synset_details(client: NLPClient):
    """Demo synset details functionality."""
    print_separator("SYNSET DETAILS DEMO")
    
    synsets = ["dog.n.01", "run.v.01", "beautiful.a.01", "quickly.r.01"]
    
    for synset_name in synsets:
        try:
            print(f"\nDetailed information for '{synset_name}':")
            result = client.get_synset_details(synset_name)
            
            synset_info = result['synset']
            print(f"  ID: {synset_info['id']}")
            print(f"  Name: {synset_info['name']}")
            print(f"  Definition: {synset_info['definition']}")
            print(f"  Part of Speech: {synset_info['pos']}")
            print(f"  Max Depth: {synset_info['max_depth']}")
            
            if synset_info['examples']:
                print(f"  Examples: {', '.join(synset_info['examples'])}")
            
            print(f"  Lemmas ({len(result['lemmas'])}):")
            for lemma in result['lemmas'][:3]:  # Show first 3
                print(f"    - {lemma['name']} (count: {lemma['count']})")
                if lemma['antonyms']:
                    print(f"      Antonyms: {', '.join(lemma['antonyms'])}")
            
            print(f"  Hypernyms ({len(result['hypernyms'])}):")
            for hypernym in result['hypernyms'][:3]:  # Show first 3
                print(f"    - {hypernym['name']}: {hypernym['definition']}")
            
            print(f"  Hyponyms ({len(result['hyponyms'])}):")
            for hyponym in result['hyponyms'][:3]:  # Show first 3
                print(f"    - {hyponym['name']}: {hyponym['definition']}")
                
        except Exception as e:
            print(f"  Error getting details for '{synset_name}': {e}")


def demo_synset_similarity(client: NLPClient):
    """Demo synset similarity calculation."""
    print_separator("SYNSET SIMILARITY DEMO")
    
    synset_pairs = [
        ("dog.n.01", "cat.n.01"),
        ("car.n.01", "automobile.n.01"),
        ("big.a.01", "large.a.01"),
        ("run.v.01", "walk.v.01"),
        ("happy.a.01", "sad.a.01")
    ]
    
    similarity_types = ["path", "wup", "lch"]
    
    for synset1, synset2 in synset_pairs:
        print(f"\nSimilarity between '{synset1}' and '{synset2}':")
        
        for sim_type in similarity_types:
            try:
                result = client.calculate_synset_similarity(synset1, synset2, sim_type)
                
                score = result['similarity_score']
                print(f"  {sim_type.upper()} similarity: {score:.4f}")
                
                if result['common_hypernyms']:
                    print(f"    Common hypernyms: {', '.join([h['name'] for h in result['common_hypernyms'][:2]])}")
                    
            except Exception as e:
                print(f"  Error calculating {sim_type} similarity: {e}")


def demo_synset_relations(client: NLPClient):
    """Demo synset relations functionality."""
    print_separator("SYNSET RELATIONS DEMO")
    
    synsets = ["dog.n.01", "vehicle.n.01", "color.n.01"]
    
    for synset_name in synsets:
        try:
            print(f"\nRelations for '{synset_name}':")
            result = client.get_synset_relations(synset_name)
            
            print(f"  Relation type: {result['relation_type']}")
            print(f"  Related synsets ({len(result['related_synsets'])}):")
            
            for synset in result['related_synsets'][:5]:  # Show first 5
                print(f"    - {synset['name']}: {synset['definition']}")
            
            if result['relation_paths']:
                print(f"  Relation paths ({len(result['relation_paths'])}):")
                for path in result['relation_paths'][:2]:  # Show first 2 paths
                    path_names = [s['name'] for s in path['path']]
                    print(f"    - Depth {path['depth']}: {' -> '.join(path_names)}")
                
        except Exception as e:
            print(f"  Error getting relations for '{synset_name}': {e}")


def demo_lemma_search(client: NLPClient):
    """Demo lemma search functionality."""
    print_separator("LEMMA SEARCH DEMO")
    
    patterns = ["dog", "run*", "*ing", "beautiful"]
    
    for pattern in patterns:
        try:
            print(f"\nSearching lemmas for pattern '{pattern}':")
            result = client.search_lemmas(pattern)
            
            print(f"  Found {len(result['lemmas'])} lemma(s)")
            
            for i, lemma in enumerate(result['lemmas'][:5]):  # Show first 5
                print(f"  {i+1}. {lemma['name']} (lang: {lemma['lang']})")
                print(f"     Key: {lemma['key']}")
                print(f"     Count: {lemma['count']}")
                if lemma['antonyms']:
                    print(f"     Antonyms: {', '.join(lemma['antonyms'])}")
                if lemma['derivationally_related_forms']:
                    print(f"     Related forms: {', '.join(lemma['derivationally_related_forms'][:3])}")
                if lemma['pertainyms']:
                    print(f"     Pertainyms: {', '.join(lemma['pertainyms'][:3])}")
                print()  # Add blank line for readability
                
        except Exception as e:
            print(f"  Error searching lemmas for '{pattern}': {e}")


def demo_synonym_search(client: NLPClient):
    """Demo synonym search functionality."""
    print_separator("SYNONYM SEARCH DEMO")
    
    words = ["dog", "big", "run", "happy", "car"]
    
    for word in words:
        try:
            print(f"\nSearching synonyms for '{word}':")
            result = client.search_synonyms(word)
            
            print(f"  Language: {result['lang']}")
            print(f"  Groups found: {len(result['synonym_groups'])}")
            
            for i, group in enumerate(result['synonym_groups'][:3]):  # Show first 3 groups
                print(f"  Group {i+1} ({group['synset_id']}):")
                print(f"    Definition: {group['sense_definition']}")
                print(f"    Synonyms: {', '.join(group['synonyms'])}")
                
        except Exception as e:
            print(f"  Error searching synonyms for '{word}': {e}")


def main():
    """Main demo function."""
    print("WordNet/Synset Endpoints Demo")
    print("=" * 60)
    
    # Initialize client
    client = NLPClient(host="localhost", port=8161)
    
    try:
        # Connect to server
        print(f"Connecting to NLP service at localhost:8161...")
        client.connect()
        print("Connected successfully!")
        
        # Wait a moment for connection to stabilize
        time.sleep(1)
        
        # Run demos
        demo_synset_lookup(client)
        demo_synset_details(client)
        demo_synset_similarity(client)
        demo_synset_relations(client)
        demo_lemma_search(client)
        demo_synonym_search(client)
        
        print_separator("DEMO COMPLETE")
        print("WordNet/Synset endpoints demo completed successfully!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        logger.error(f"Demo failed: {e}")
        return 1
        
    finally:
        # Cleanup
        try:
            client.disconnect()
        except:
            pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
