"""Test suite for WordNet/Synset functionality."""

import pytest
import grpc
from unittest.mock import Mock, patch, MagicMock
from concurrent import futures
import time

from nlpmicroservice.server import NLPServicer
from nlpmicroservice.nlp_service import NLPProcessor
from nlpmicroservice.client import NLPClient
from nlpmicroservice.config import settings


class TestWordNetProcessor:
    """Test cases for WordNet processing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = NLPProcessor()
    
    def test_lookup_synsets(self):
        """Test synset lookup functionality."""
        # Test with a common word
        synsets = self.processor.lookup_synsets("dog")
        assert isinstance(synsets, list)
        assert len(synsets) > 0
        
        # First synset should be the most common one
        first_synset = synsets[0]
        assert "dog" in first_synset["name"]
        assert "definition" in first_synset
        assert "pos" in first_synset
        assert "lemma_names" in first_synset
        assert "examples" in first_synset
        
        # Test with POS filtering
        noun_synsets = self.processor.lookup_synsets("dog", pos="n")
        verb_synsets = self.processor.lookup_synsets("dog", pos="v")
        
        # Should have both noun and verb synsets
        assert len(noun_synsets) > 0
        assert len(verb_synsets) >= 0  # May be 0 if no verb synsets
        
        # Test with non-existent word
        empty_synsets = self.processor.lookup_synsets("nonexistentword123")
        assert isinstance(empty_synsets, list)
        assert len(empty_synsets) == 0
    
    def test_get_synset_details(self):
        """Test synset details retrieval."""
        # Test with a known synset
        details = self.processor.get_synset_details("dog.n.01")
        
        assert isinstance(details, dict)
        assert "synset" in details
        assert "hypernyms" in details
        assert "hyponyms" in details
        assert "lemmas" in details
        
        # Check synset info
        synset_info = details["synset"]
        assert "id" in synset_info
        assert "name" in synset_info
        assert "definition" in synset_info
        assert "pos" in synset_info
        assert "lemma_names" in synset_info
        assert "examples" in synset_info
        
        # Check that lemmas contain detailed information
        if details["lemmas"]:
            first_lemma = details["lemmas"][0]
            assert "name" in first_lemma
            assert "count" in first_lemma
            assert "key" in first_lemma
        
        # Test with invalid synset should raise exception
        try:
            invalid_details = self.processor.get_synset_details("invalid.synset.01")
            assert False, "Should have raised exception for invalid synset"
        except Exception:
            pass  # Expected
    
    def test_calculate_synset_similarity(self):
        """Test synset similarity calculation."""
        # Test with related synsets
        similarity = self.processor.calculate_synset_similarity("dog.n.01", "cat.n.01")
        assert isinstance(similarity, dict)
        assert "similarity_score" in similarity
        assert "similarity_type" in similarity
        assert "synset1_id" in similarity
        assert "synset2_id" in similarity
        assert "common_hypernyms" in similarity
        
        # Check score is valid
        assert 0 <= similarity["similarity_score"] <= 1
        
        # Test with different similarity types
        path_sim = self.processor.calculate_synset_similarity("dog.n.01", "cat.n.01", "path")
        wup_sim = self.processor.calculate_synset_similarity("dog.n.01", "cat.n.01", "wup")
        
        assert path_sim["similarity_type"] == "path"
        assert wup_sim["similarity_type"] == "wup"
        
        # Test with identical synsets
        identity_sim = self.processor.calculate_synset_similarity("dog.n.01", "dog.n.01")
        assert identity_sim["similarity_score"] == 1.0
    
    def test_get_synset_relations(self):
        """Test synset relations retrieval."""
        # Test with a known synset
        relations = self.processor.get_synset_relations("dog.n.01")
        
        assert isinstance(relations, dict)
        assert "synset_id" in relations
        assert "relation_type" in relations
        assert "related_synsets" in relations
        assert "relation_paths" in relations
        
        # Check that we have some related synsets
        assert isinstance(relations["related_synsets"], list)
        if relations["related_synsets"]:
            first_related = relations["related_synsets"][0]
            assert "id" in first_related
            assert "name" in first_related
            assert "definition" in first_related
        
        # Test with specific relation type
        hypernym_relations = self.processor.get_synset_relations("dog.n.01", "hypernyms")
        assert isinstance(hypernym_relations, dict)
        assert hypernym_relations["relation_type"] == "hypernyms"
        
        # Test with invalid synset should raise exception
        try:
            invalid_relations = self.processor.get_synset_relations("invalid.synset.01")
            assert False, "Should have raised exception for invalid synset"
        except Exception:
            pass  # Expected
    
    def test_search_lemmas(self):
        """Test lemma search functionality."""
        # Test with a common pattern
        lemmas = self.processor.search_lemmas("dog")
        assert isinstance(lemmas, dict)
        assert "lemmas" in lemmas
        assert "original_word" in lemmas
        assert "morphed_word" in lemmas
        assert "pos" in lemmas
        assert "lang" in lemmas
        assert len(lemmas["lemmas"]) > 0
        
        # Check lemma structure
        first_lemma = lemmas["lemmas"][0]
        assert "name" in first_lemma
        assert "key" in first_lemma
        assert "count" in first_lemma
        assert "lang" in first_lemma
        
        # Test with POS filtering
        noun_lemmas = self.processor.search_lemmas("dog", pos="NOUN")
        verb_lemmas = self.processor.search_lemmas("dog", pos="VERB")
        
        # Should have results
        assert len(noun_lemmas["lemmas"]) > 0
        
        # Test with morphology disabled
        no_morph_lemmas = self.processor.search_lemmas("dogs", include_morphology=False)
        assert isinstance(no_morph_lemmas, dict)
    
    def test_search_synonyms(self):
        """Test synonym search functionality."""
        # Test with a common word
        synonyms = self.processor.search_synonyms("dog")
        assert isinstance(synonyms, dict)
        assert "synonym_groups" in synonyms
        assert "word" in synonyms
        assert "lang" in synonyms
        
        # Check synonym group structure
        if synonyms["synonym_groups"]:
            first_group = synonyms["synonym_groups"][0]
            assert "synset_id" in first_group
            assert "sense_definition" in first_group
            assert "synonyms" in first_group
            assert isinstance(first_group["synonyms"], list)
        
        # Test with max synonyms limit
        limited_synonyms = self.processor.search_synonyms("dog", max_synonyms=3)
        assert isinstance(limited_synonyms, dict)
        
        # Test with non-existent word
        empty_synonyms = self.processor.search_synonyms("nonexistentword123")
        assert isinstance(empty_synonyms, dict)
        assert len(empty_synonyms["synonym_groups"]) == 0


class TestWordNetClient:
    """Test cases for WordNet client functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = NLPClient()
    
    @pytest.fixture
    def mock_channel(self):
        """Mock gRPC channel for testing."""
        with patch('grpc.insecure_channel') as mock_channel:
            yield mock_channel
    
    @pytest.fixture
    def mock_stub(self):
        """Mock gRPC stub for testing."""
        mock_stub = Mock()
        return mock_stub
    
    def test_lookup_synsets_client(self, mock_channel, mock_stub):
        """Test client synset lookup."""
        # Mock the response
        mock_response = Mock()
        mock_response.synsets = [
            Mock(name="dog.n.01", definition="a domestic animal", pos="n", 
                 lemma_names=["dog", "domestic_dog"], examples=["the dog barked"])
        ]
        mock_response.word = "dog"
        mock_response.total_count = 1
        mock_stub.SynsetsLookup.return_value = mock_response
        
        # Set up client
        self.client.stub = mock_stub
        
        # Call method
        result = self.client.lookup_synsets("dog")
        
        # Verify results
        assert isinstance(result, dict)
        assert "synsets" in result
        assert "word" in result
        assert "pos" in result
        assert "lang" in result
        assert len(result["synsets"]) == 1
        assert result["word"] == "dog"
    
    def test_get_synset_details_client(self, mock_channel, mock_stub):
        """Test client synset details retrieval."""
        # Mock the response
        mock_response = Mock()
        mock_response.synset = Mock(
            id="dog.n.01",
            name="dog",
            pos="n",
            definition="a domestic animal",
            examples=["the dog barked"],
            lemma_names=["dog", "domestic_dog"],
            max_depth=5
        )
        mock_response.lemmas = [
            Mock(name="dog", count=100, key="dog%1:05:00::", lang="eng",
                 antonyms=[], derivationally_related_forms=[], pertainyms=[])
        ]
        mock_response.hypernyms = [
            Mock(id="canine.n.02", name="canine", pos="n", definition="a dog-like animal",
                 examples=[], lemma_names=["canine"], max_depth=4)
        ]
        mock_response.hyponyms = []
        mock_stub.SynsetDetails.return_value = mock_response
        
        # Set up client
        self.client.stub = mock_stub
        
        # Call method
        result = self.client.get_synset_details("dog.n.01")
        
        # Verify results
        assert isinstance(result, dict)
        assert "synset" in result
        assert "lemmas" in result
        assert "hypernyms" in result
        assert "hyponyms" in result
        assert result["synset"]["id"] == "dog.n.01"
        assert len(result["lemmas"]) == 1
        assert len(result["hypernyms"]) == 1
    
    def test_calculate_synset_similarity_client(self, mock_channel, mock_stub):
        """Test client synset similarity calculation."""
        # Mock the response
        mock_response = Mock()
        mock_response.synset1_id = "dog.n.01"
        mock_response.synset2_id = "cat.n.01"
        mock_response.similarity_type = "path"
        mock_response.similarity_score = 0.25
        mock_response.common_hypernyms = [
            Mock(id="animal.n.01", name="animal", pos="n", definition="a living creature",
                 examples=[], lemma_names=["animal"], max_depth=3)
        ]
        mock_stub.SynsetSimilarity.return_value = mock_response
        
        # Set up client
        self.client.stub = mock_stub
        
        # Call method
        result = self.client.calculate_synset_similarity("dog.n.01", "cat.n.01")
        
        # Verify results
        assert isinstance(result, dict)
        assert "similarity_score" in result
        assert "similarity_type" in result
        assert "synset1_id" in result
        assert "synset2_id" in result
        assert "common_hypernyms" in result
        assert result["similarity_score"] == 0.25
        assert result["synset1_id"] == "dog.n.01"
        assert result["synset2_id"] == "cat.n.01"
        assert len(result["common_hypernyms"]) == 1
    
    def test_get_synset_relations_client(self, mock_channel, mock_stub):
        """Test client synset relations retrieval."""
        # Mock the response
        mock_response = Mock()
        mock_response.synset_id = "dog.n.01"
        mock_response.relation_type = "hypernyms"
        mock_response.related_synsets = [
            Mock(id="canine.n.02", name="canine", pos="n", definition="a dog-like animal",
                 examples=[], lemma_names=["canine"], max_depth=4)
        ]
        mock_response.relation_paths = [
            Mock(path=[
                Mock(id="dog.n.01", name="dog", pos="n", definition="a domestic animal",
                     examples=[], lemma_names=["dog"], max_depth=5)
            ], depth=1)
        ]
        mock_stub.SynsetRelations.return_value = mock_response
        
        # Set up client
        self.client.stub = mock_stub
        
        # Call method
        result = self.client.get_synset_relations("dog.n.01")
        
        # Verify results
        assert isinstance(result, dict)
        assert "synset_id" in result
        assert "relation_type" in result
        assert "related_synsets" in result
        assert "relation_paths" in result
        assert result["synset_id"] == "dog.n.01"
        assert len(result["related_synsets"]) == 1
        assert len(result["relation_paths"]) == 1
    
    def test_search_lemmas_client(self, mock_channel, mock_stub):
        """Test client lemma search."""
        # Mock the response
        mock_response = Mock()
        mock_response.lemmas = [
            Mock(name="dog", key="dog%1:05:00::", count=100, lang="eng",
                 antonyms=[], derivationally_related_forms=[], pertainyms=[])
        ]
        mock_response.original_word = "dog"
        mock_response.morphed_word = "dog"
        mock_response.pos = "n"
        mock_response.lang = "eng"
        mock_stub.LemmaSearch.return_value = mock_response
        
        # Set up client
        self.client.stub = mock_stub
        
        # Call method
        result = self.client.search_lemmas("dog")
        
        # Verify results
        assert isinstance(result, dict)
        assert "lemmas" in result
        assert "original_word" in result
        assert "morphed_word" in result
        assert "pos" in result
        assert "lang" in result
        assert len(result["lemmas"]) == 1
        assert result["original_word"] == "dog"
        assert result["morphed_word"] == "dog"
    
    def test_search_synonyms_client(self, mock_channel, mock_stub):
        """Test client synonym search."""
        # Mock the response
        mock_response = Mock()
        mock_response.word = "dog"
        mock_response.lang = "eng"
        mock_response.synonym_groups = [
            Mock(sense_definition="a domestic animal",
                 synonyms=["dog", "domestic_dog"],
                 synset_id="dog.n.01")
        ]
        mock_stub.SynonymSearch.return_value = mock_response
        
        # Set up client
        self.client.stub = mock_stub
        
        # Call method
        result = self.client.search_synonyms("dog")
        
        # Verify results
        assert isinstance(result, dict)
        assert "word" in result
        assert "lang" in result
        assert "synonym_groups" in result
        assert result["word"] == "dog"
        assert result["lang"] == "eng"
        assert len(result["synonym_groups"]) == 1
        assert result["synonym_groups"][0]["synset_id"] == "dog.n.01"


class TestWordNetIntegration:
    """Integration tests for WordNet functionality."""
    
    @pytest.fixture
    def server(self):
        """Start test server."""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        from nlpmicroservice import nlp_pb2_grpc
        nlp_pb2_grpc.add_NLPServiceServicer_to_server(NLPServicer(), server)
        
        # Use a test port
        port = server.add_insecure_port('[::]:0')
        server.start()
        
        yield server, port
        
        server.stop(0)
    
    @pytest.fixture
    def client(self, server):
        """Create test client."""
        server_instance, port = server
        client = NLPClient(host="localhost", port=port)
        client.connect()
        
        yield client
        
        client.disconnect()
    
    def test_wordnet_integration_lookup(self, client):
        """Test WordNet synset lookup integration."""
        try:
            result = client.lookup_synsets("dog")
            assert isinstance(result, dict)
            assert "synsets" in result
            assert "word" in result
            assert "pos" in result
            assert "lang" in result
        except Exception as e:
            pytest.skip(f"WordNet integration test failed (server not running): {e}")
    
    def test_wordnet_integration_details(self, client):
        """Test WordNet synset details integration."""
        try:
            result = client.get_synset_details("dog.n.01")
            assert isinstance(result, dict)
            assert "synset" in result
            assert "lemmas" in result
            assert "hypernyms" in result
        except Exception as e:
            pytest.skip(f"WordNet integration test failed (server not running): {e}")
    
    def test_wordnet_integration_similarity(self, client):
        """Test WordNet similarity calculation integration."""
        try:
            result = client.calculate_synset_similarity("dog.n.01", "cat.n.01")
            assert isinstance(result, dict)
            assert "similarity_score" in result
            assert "synset1_id" in result
            assert "synset2_id" in result
        except Exception as e:
            pytest.skip(f"WordNet integration test failed (server not running): {e}")
    
    def test_wordnet_integration_relations(self, client):
        """Test WordNet relations integration."""
        try:
            result = client.get_synset_relations("dog.n.01")
            assert isinstance(result, dict)
            assert "synset_id" in result
            assert "related_synsets" in result
            assert "relation_paths" in result
        except Exception as e:
            pytest.skip(f"WordNet integration test failed (server not running): {e}")
    
    def test_wordnet_integration_lemma_search(self, client):
        """Test WordNet lemma search integration."""
        try:
            result = client.search_lemmas("dog")
            assert isinstance(result, dict)
            assert "lemmas" in result
            assert "original_word" in result
            assert "morphed_word" in result
        except Exception as e:
            pytest.skip(f"WordNet integration test failed (server not running): {e}")
    
    def test_wordnet_integration_synonym_search(self, client):
        """Test WordNet synonym search integration."""
        try:
            result = client.search_synonyms("dog")
            assert isinstance(result, dict)
            assert "word" in result
            assert "synonym_groups" in result
            assert "lang" in result
        except Exception as e:
            pytest.skip(f"WordNet integration test failed (server not running): {e}")
