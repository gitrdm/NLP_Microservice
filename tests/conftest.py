"""Test configuration and fixtures."""

import pytest
import os
import tempfile


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Natural Language Processing (NLP) is a fascinating field that combines 
    computer science, artificial intelligence, and linguistics. It enables 
    computers to understand, interpret, and generate human language in a 
    valuable way. Companies like Google, Microsoft, and Apple use NLP 
    technologies in their products.
    """


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def nlp_processor():
    """Create an NLP processor instance."""
    from nlpmicroservice.nlp_service import NLPProcessor
    return NLPProcessor()
