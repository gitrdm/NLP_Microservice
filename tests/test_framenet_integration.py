"""
Integration tests for FrameNet endpoints.
"""

import pytest
import time
from src.nlpmicroservice.client import NLPClient
from src.nlpmicroservice.server import serve
import threading
import os
import signal


class TestFrameNetIntegration:
    """Integration tests for FrameNet functionality."""

    @pytest.fixture(scope="class")
    def server_thread(self):
        """Start server in a separate thread."""
        def start_server():
            serve()
        
        thread = threading.Thread(target=start_server, daemon=True)
        thread.start()
        time.sleep(2)  # Give server time to start
        yield
        # Server will be killed by daemon thread cleanup

    @pytest.fixture
    def client(self, server_thread):
        """Create and connect client."""
        client = NLPClient()
        client.connect()
        yield client
        client.disconnect()

    def test_frame_search(self, client):
        """Test frame search functionality."""
        result = client.search_frames("communication", max_results=5)
        frames = result["frames"]
        assert len(frames) > 0
        assert any("communication" in frame["name"].lower() for frame in frames)
        
        # Check frame structure
        first_frame = frames[0]
        assert "id" in first_frame
        assert "name" in first_frame
        assert "definition" in first_frame
        assert "lexical_units" in first_frame

    def test_frame_details(self, client):
        """Test frame details functionality."""
        details = client.get_frame_details(frame_name="Communication")
        
        assert "id" in details
        assert "name" in details
        assert details["name"] == "Communication"
        assert "definition" in details
        assert "lexical_units" in details
        assert "frame_elements" in details
        assert "frame_relations" in details
        
        # Check that we have some lexical units
        assert len(details["lexical_units"]) > 0
        
        # Check that we have some frame elements
        assert len(details["frame_elements"]) > 0

    def test_lexical_unit_search(self, client):
        """Test lexical unit search functionality."""
        result = client.search_lexical_units("speak", max_results=10)
        lus = result["lexical_units"]
        
        assert len(lus) > 0
        assert any("speak" in lu["name"] for lu in lus)
        
        # Check LU structure
        first_lu = lus[0]
        assert "id" in first_lu
        assert "name" in first_lu
        assert "pos" in first_lu
        assert "frame_name" in first_lu
        assert "frame_id" in first_lu

    def test_frame_relations(self, client):
        """Test frame relations functionality."""
        relations = client.get_frame_relations(frame_name="Communication")
        
        assert "relations" in relations
        assert "relation_types" in relations
        
        # Should have some relation types
        assert len(relations["relation_types"]) > 0
        assert "Inheritance" in relations["relation_types"]

    def test_semantic_role_labeling(self, client):
        """Test semantic role labeling functionality."""
        text = "John told Mary about the meeting."
        frames = client.semantic_role_labeling(text)
        
        assert len(frames) > 0
        
        # Check frame structure
        first_frame = frames[0]
        assert "frame_name" in first_frame
        assert "frame_id" in first_frame
        assert "trigger_word" in first_frame
        assert "roles" in first_frame

    def test_performance_basic(self, client):
        """Test basic performance characteristics."""
        # Test that operations complete in reasonable time
        start_time = time.time()
        
        # Frame search should be fast
        frames = client.search_frames("communication", max_results=5)
        search_time = time.time() - start_time
        assert search_time < 1.0  # Should complete within 1 second
        
        # Frame details should be reasonable
        start_time = time.time()
        details = client.get_frame_details(frame_name="Communication")
        details_time = time.time() - start_time
        assert details_time < 2.0  # Should complete within 2 seconds
        
        # SRL should be reasonable for short text
        start_time = time.time()
        frames = client.semantic_role_labeling("John spoke.")
        srl_time = time.time() - start_time
        assert srl_time < 3.0  # Should complete within 3 seconds

    def test_error_handling(self, client):
        """Test error handling in FrameNet endpoints."""
        # Test invalid frame name
        with pytest.raises(Exception):
            client.get_frame_details(frame_name="NonExistentFrame")
        
        # Test invalid frame ID
        with pytest.raises(Exception):
            client.get_frame_details(frame_id=99999)
        
        # Test empty search
        result = client.search_frames("xyzabc123nonexistent", max_results=5)
        assert len(result["frames"]) == 0

    def test_framenet_data_availability(self, client):
        """Test that FrameNet data is properly available."""
        # Search for some common frames
        common_frames = ["Communication", "Motion", "Cause_motion"]
        
        for frame_name in common_frames:
            try:
                details = client.get_frame_details(frame_name=frame_name)
                assert details["name"] == frame_name
                assert len(details["lexical_units"]) > 0
            except Exception as e:
                pytest.fail(f"Failed to get details for common frame '{frame_name}': {e}")
