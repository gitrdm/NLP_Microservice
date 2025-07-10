# WordNet API Documentation

This document describes the WordNet/Synset endpoints available in the NLP microservice, providing comprehensive access to WordNet's semantic network.

## Overview

WordNet is a large lexical database of English words organized into sets of synonyms (synsets). The WordNet endpoints in this microservice provide access to:

- **Synset Lookup**: Find synsets for words
- **Synset Details**: Get comprehensive information about synsets
- **Similarity Calculation**: Compute semantic similarity between synsets
- **Relation Exploration**: Navigate WordNet's semantic relations
- **Lemma Search**: Search for lemmas with pattern matching
- **Synonym Discovery**: Find synonyms for words

## API Endpoints

### 1. SynsetsLookup

**Purpose**: Look up synsets for a given word.

**Client Method**: `lookup_synsets(word, pos=None, lang="eng")`

**Parameters**:
- `word` (str): The word to look up
- `pos` (str, optional): Part of speech filter ("n", "v", "a", "r", "s")
- `lang` (str, optional): Language code (default: "eng")

**Returns**:
```python
{
    "synsets": [
        {
            "name": "dog.n.01",
            "definition": "a member of the genus Canis...",
            "pos": "n",
            "lemma_names": ["dog", "domestic_dog", "Canis_familiaris"],
            "examples": ["the dog barked all night"]
        }
    ],
    "word": "dog",
    "total_count": 8
}
```

**Example Usage**:
```python
# Look up all synsets for "dog"
result = client.lookup_synsets("dog")

# Look up only noun synsets
result = client.lookup_synsets("dog", pos="n")
```

### 2. SynsetDetails

**Purpose**: Get detailed information about a specific synset.

**Client Method**: `get_synset_details(synset_name, include_relations=True, include_examples=True)`

**Parameters**:
- `synset_name` (str): The synset name (e.g., "dog.n.01")
- `include_relations` (bool, optional): Include relation information
- `include_examples` (bool, optional): Include example sentences

**Returns**:
```python
{
    "name": "dog.n.01",
    "definition": "a member of the genus Canis...",
    "pos": "n",
    "offset": 2084071,
    "examples": ["the dog barked all night"],
    "lemma_names": ["dog", "domestic_dog", "Canis_familiaris"],
    "lemmas": [
        {
            "name": "dog",
            "count": 61,
            "key": "dog%1:05:00::",
            "antonyms": [],
            "derivationally_related_forms": ["dog.v.01"],
            "pertainyms": []
        }
    ],
    "relations": [
        {
            "type": "hypernyms",
            "synsets": ["canine.n.02", "domestic_animal.n.01"]
        }
    ],
    "max_depth": 14,
    "min_depth": 14
}
```

**Example Usage**:
```python
# Get full details for a synset
result = client.get_synset_details("dog.n.01")

# Get details without relations
result = client.get_synset_details("dog.n.01", include_relations=False)
```

### 3. SynsetSimilarity

**Purpose**: Calculate semantic similarity between two synsets.

**Client Method**: `calculate_synset_similarity(synset1, synset2, similarity_type="path")`

**Parameters**:
- `synset1` (str): First synset name
- `synset2` (str): Second synset name
- `similarity_type` (str): Type of similarity ("path", "lch", "wup", "res", "jcn", "lin")

**Returns**:
```python
{
    "synset1": "dog.n.01",
    "synset2": "cat.n.01",
    "similarity_type": "path",
    "similarity_score": 0.2,
    "valid": True,
    "error_message": ""
}
```

**Similarity Types**:
- `path`: Path similarity (0-1, based on shortest path)
- `lch`: Leacock-Chodorow similarity (based on path length and taxonomy depth)
- `wup`: Wu-Palmer similarity (based on depths and least common subsumer)
- `res`: Resnik similarity (requires information content)
- `jcn`: Jiang-Conrath similarity (requires information content)
- `lin`: Lin similarity (requires information content)

**Example Usage**:
```python
# Calculate path similarity
result = client.calculate_synset_similarity("dog.n.01", "cat.n.01")

# Calculate Wu-Palmer similarity
result = client.calculate_synset_similarity("dog.n.01", "cat.n.01", "wup")
```

### 4. SynsetRelations

**Purpose**: Get semantic relations for a synset.

**Client Method**: `get_synset_relations(synset_name, relation_type=None, max_depth=1)`

**Parameters**:
- `synset_name` (str): The synset name
- `relation_type` (str, optional): Filter by relation type
- `max_depth` (int, optional): Maximum depth for traversal

**Returns**:
```python
{
    "synset_name": "dog.n.01",
    "relations": [
        {
            "type": "hypernyms",
            "synsets": ["canine.n.02", "domestic_animal.n.01"]
        },
        {
            "type": "hyponyms",
            "synsets": ["basenji.n.01", "corgi.n.01", "cur.n.01"]
        }
    ],
    "available_relation_types": ["hypernyms", "hyponyms", "meronyms", "holonyms"]
}
```

**Available Relation Types**:
- `hypernyms`: More general concepts
- `hyponyms`: More specific concepts
- `meronyms`: Parts of the concept
- `holonyms`: Wholes that contain the concept
- `similar_tos`: Similar concepts
- `also_sees`: Related concepts
- `entailments`: Entailed actions (for verbs)
- `causes`: Caused actions (for verbs)
- `attributes`: Associated attributes
- `derivationally_related_forms`: Related word forms

**Example Usage**:
```python
# Get all relations
result = client.get_synset_relations("dog.n.01")

# Get only hypernyms
result = client.get_synset_relations("dog.n.01", relation_type="hypernyms")
```

### 5. LemmaSearch

**Purpose**: Search for lemmas using pattern matching.

**Client Method**: `search_lemmas(lemma_pattern, pos=None, lang="eng", max_results=50)`

**Parameters**:
- `lemma_pattern` (str): Pattern to search for (supports wildcards)
- `pos` (str, optional): Part of speech filter
- `lang` (str, optional): Language code
- `max_results` (int, optional): Maximum number of results

**Returns**:
```python
{
    "lemmas": [
        {
            "name": "dog",
            "synset_name": "dog.n.01",
            "pos": "n",
            "definition": "a member of the genus Canis...",
            "count": 61,
            "key": "dog%1:05:00::"
        }
    ],
    "pattern": "dog",
    "total_count": 6
}
```

**Example Usage**:
```python
# Search for exact lemma
result = client.search_lemmas("dog")

# Search with wildcards
result = client.search_lemmas("dog*")

# Search with POS filter
result = client.search_lemmas("run", pos="v")
```

### 6. SynonymSearch

**Purpose**: Find synonyms for a word.

**Client Method**: `search_synonyms(word, pos=None, lang="eng", include_definitions=False)`

**Parameters**:
- `word` (str): The word to find synonyms for
- `pos` (str, optional): Part of speech filter
- `lang` (str, optional): Language code
- `include_definitions` (bool, optional): Include definitions for synonyms

**Returns**:
```python
{
    "word": "dog",
    "synonym_groups": [
        {
            "synset_name": "dog.n.01",
            "definition": "a member of the genus Canis...",
            "synonyms": [
                {
                    "word": "dog",
                    "definition": "a member of the genus Canis..."
                },
                {
                    "word": "domestic_dog",
                    "definition": "a member of the genus Canis..."
                }
            ]
        }
    ],
    "total_synonyms": 8
}
```

**Example Usage**:
```python
# Find synonyms for a word
result = client.search_synonyms("dog")

# Find synonyms with definitions
result = client.search_synonyms("dog", include_definitions=True)

# Find synonyms for adjectives only
result = client.search_synonyms("big", pos="a")
```

## Usage Examples

### Basic Synset Exploration

```python
from nlpmicroservice.client import NLPClient

# Connect to service
client = NLPClient()
client.connect()

# Look up synsets for "dog"
synsets = client.lookup_synsets("dog")
print(f"Found {synsets['total_count']} synsets for 'dog'")

# Get details for the first synset
first_synset = synsets['synsets'][0]
details = client.get_synset_details(first_synset['name'])
print(f"Definition: {details['definition']}")
print(f"Examples: {details['examples']}")

# Get hypernyms (more general concepts)
relations = client.get_synset_relations(first_synset['name'], "hypernyms")
for relation in relations['relations']:
    if relation['type'] == 'hypernyms':
        print(f"Hypernyms: {relation['synsets']}")

client.disconnect()
```

### Similarity Analysis

```python
# Compare similarity between related concepts
similarity = client.calculate_synset_similarity("dog.n.01", "cat.n.01")
print(f"Dog-Cat similarity: {similarity['similarity_score']:.3f}")

# Compare different similarity metrics
metrics = ["path", "wup", "lch"]
for metric in metrics:
    sim = client.calculate_synset_similarity("dog.n.01", "cat.n.01", metric)
    if sim['valid']:
        print(f"{metric.upper()}: {sim['similarity_score']:.3f}")
```

### Synonym Discovery

```python
# Find synonyms for "big"
synonyms = client.search_synonyms("big")
print(f"Found {synonyms['total_synonyms']} synonyms for 'big'")

for group in synonyms['synonym_groups']:
    print(f"\nSynset: {group['synset_name']}")
    print(f"Definition: {group['definition']}")
    synonym_words = [s['word'] for s in group['synonyms']]
    print(f"Synonyms: {', '.join(synonym_words)}")
```

### Advanced Pattern Matching

```python
# Search for lemmas ending with "ing"
lemmas = client.search_lemmas("*ing", pos="v")
print(f"Found {len(lemmas['lemmas'])} verb lemmas ending with 'ing'")

# Search for lemmas starting with "run"
lemmas = client.search_lemmas("run*")
for lemma in lemmas['lemmas'][:5]:  # Show first 5
    print(f"{lemma['name']} ({lemma['pos']}) - {lemma['synset_name']}")
```

## Error Handling

All WordNet endpoints include proper error handling:

```python
try:
    result = client.lookup_synsets("nonexistentword")
    if result['total_count'] == 0:
        print("No synsets found for this word")
except Exception as e:
    print(f"Error: {e}")
```

## Multilingual Support

WordNet endpoints support multiple languages through the Open Multilingual WordNet:

```python
# Look up synsets in Japanese
result = client.lookup_synsets("犬", lang="jpn")

# Search for lemmas in Spanish
result = client.search_lemmas("perro", lang="spa")
```

## Performance Considerations

- **Caching**: Results are cached for frequently accessed synsets
- **Batch Operations**: Use pattern matching for bulk operations
- **Depth Limits**: Set appropriate max_depth for relation traversal
- **Result Limits**: Use max_results parameter to control response size

## Integration with Other Endpoints

WordNet endpoints can be combined with other NLP endpoints:

```python
# Analyze sentiment of synset definitions
synset_details = client.get_synset_details("happy.a.01")
sentiment = client.analyze_sentiment(synset_details['definition'])
print(f"Sentiment of 'happy' definition: {sentiment['sentiment']}")

# Extract keywords from synset examples
for example in synset_details['examples']:
    keywords = client.extract_keywords(example)
    print(f"Keywords in example: {[k['word'] for k in keywords['keywords']]}")
```

## Demo Script

A comprehensive demo script is available at `example_files/wordnet_demo.py`:

```bash
python example_files/wordnet_demo.py
```

This script demonstrates all WordNet endpoints with real examples and provides a good starting point for understanding the API.

## References

- [WordNet Official Site](https://wordnet.princeton.edu/)
- [NLTK WordNet Interface](https://www.nltk.org/howto/wordnet.html)
- [Open Multilingual WordNet](http://compling.hss.ntu.edu.sg/omw/)
- [WordNet Similarity Measures](https://www.nltk.org/howto/wordnet.html#similarity)
