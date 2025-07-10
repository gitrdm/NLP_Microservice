"""Client for testing the NLP gRPC service."""

import grpc
from typing import List, Dict, Any
from loguru import logger

from .config import settings

# Import generated gRPC modules (will be generated from proto files)
try:
    from . import nlp_pb2
    from . import nlp_pb2_grpc
except ImportError:
    logger.warning("Generated gRPC modules not found. Run 'make generate-proto' to generate them.")
    # Create placeholder classes for development
    class nlp_pb2:
        pass
    class nlp_pb2_grpc:
        pass


class NLPClient:
    """Client for the NLP gRPC service."""
    
    def __init__(self, host: str = "localhost", port: int = 8161):
        """Initialize the NLP client."""
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None
    
    def connect(self):
        """Connect to the gRPC server."""
        self.channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        self.stub = nlp_pb2_grpc.NLPServiceStub(self.channel)
        logger.info(f"Connected to NLP service at {self.host}:{self.port}")
    
    def disconnect(self):
        """Disconnect from the gRPC server."""
        if self.channel:
            self.channel.close()
            logger.info("Disconnected from NLP service")
    
    def tokenize(self, text: str, language: str = "english") -> Dict[str, Any]:
        """Tokenize text."""
        if not self.stub:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        request = nlp_pb2.TokenizeRequest()
        request.text = text
        request.language = language
        
        response = self.stub.Tokenize(request)
        return {
            "tokens": list(response.tokens),
            "token_count": response.token_count
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        if not self.stub:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        request = nlp_pb2.SentimentRequest()
        request.text = text
        
        response = self.stub.SentimentAnalysis(request)
        return {
            "sentiment": response.sentiment,
            "confidence": response.confidence
        }
    
    def extract_named_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        if not self.stub:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        request = nlp_pb2.NERRequest()
        request.text = text
        
        response = self.stub.NamedEntityRecognition(request)
        return [
            {
                "text": entity.text,
                "label": entity.label,
                "start": entity.start,
                "end": entity.end
            }
            for entity in response.entities
        ]
    
    def pos_tagging(self, text: str) -> List[Dict[str, str]]:
        """Perform part-of-speech tagging."""
        if not self.stub:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        request = nlp_pb2.POSRequest()
        request.text = text
        
        response = self.stub.POSTagging(request)
        return [
            {
                "word": tag.word,
                "tag": tag.tag
            }
            for tag in response.tags
        ]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity."""
        if not self.stub:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        request = nlp_pb2.SimilarityRequest()
        request.text1 = text1
        request.text2 = text2
        
        response = self.stub.TextSimilarity(request)
        return response.similarity
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[Dict[str, Any]]:
        """Extract keywords from text."""
        if not self.stub:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        request = nlp_pb2.KeywordRequest()
        request.text = text
        request.max_keywords = max_keywords
        
        response = self.stub.KeywordExtraction(request)
        return [
            {
                "word": keyword.word,
                "score": keyword.score
            }
            for keyword in response.keywords
        ]
    
    def summarize_text(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
        """Summarize text using extractive summarization."""
        try:
            request = nlp_pb2.SummarizationRequest(text=text, max_sentences=max_sentences)
            response = self.stub.TextSummarization(request)
            
            return {
                'summary': response.summary,
                'original_sentence_count': response.original_sentence_count,
                'summary_sentence_count': response.summary_sentence_count
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC error in summarize_text: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in summarize_text: {e}")
            raise

    def search_frames(self, name_pattern: str = None, max_results: int = 50) -> Dict[str, Any]:
        """Search for frames by name pattern."""
        try:
            request = nlp_pb2.FrameSearchRequest(
                name_pattern=name_pattern or "",
                max_results=max_results
            )
            response = self.stub.FrameSearch(request)
            
            frames = []
            for frame_info in response.frames:
                frames.append({
                    'id': frame_info.id,
                    'name': frame_info.name,
                    'definition': frame_info.definition,
                    'lexical_units': list(frame_info.lexical_units)
                })
            
            return {
                'frames': frames,
                'total_count': response.total_count
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC error in search_frames: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in search_frames: {e}")
            raise

    def get_frame_details(self, frame_id: int = None, frame_name: str = None) -> Dict[str, Any]:
        """Get detailed information about a specific frame."""
        try:
            if frame_id:
                request = nlp_pb2.FrameDetailsRequest(frame_id=frame_id)
            elif frame_name:
                request = nlp_pb2.FrameDetailsRequest(frame_name=frame_name)
            else:
                raise ValueError("Either frame_id or frame_name must be provided")
            
            response = self.stub.FrameDetails(request)
            
            # Convert lexical units
            lexical_units = []
            for lu in response.lexical_units:
                lexical_units.append({
                    'id': lu.id,
                    'name': lu.name,
                    'pos': lu.pos,
                    'definition': lu.definition,
                    'status': lu.status
                })
            
            # Convert frame elements
            frame_elements = []
            for fe in response.frame_elements:
                frame_elements.append({
                    'id': fe.id,
                    'name': fe.name,
                    'definition': fe.definition,
                    'core_type': fe.core_type,
                    'abbreviation': fe.abbreviation
                })
            
            # Convert frame relations
            frame_relations = []
            for rel in response.frame_relations:
                frame_relations.append({
                    'id': rel.id,
                    'type': rel.type,
                    'parent_frame': rel.parent_frame,
                    'child_frame': rel.child_frame,
                    'description': rel.description
                })
            
            # Convert semantic types
            semantic_types = []
            for st in response.semantic_types:
                semantic_types.append({
                    'id': st.id,
                    'name': st.name,
                    'abbreviation': st.abbreviation,
                    'definition': st.definition
                })
            
            return {
                'id': response.id,
                'name': response.name,
                'definition': response.definition,
                'lexical_units': lexical_units,
                'frame_elements': frame_elements,
                'frame_relations': frame_relations,
                'semantic_types': semantic_types
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC error in get_frame_details: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in get_frame_details: {e}")
            raise

    def search_lexical_units(self, name_pattern: str = None, frame_pattern: str = None, 
                           max_results: int = 50) -> Dict[str, Any]:
        """Search for lexical units."""
        try:
            request = nlp_pb2.LexicalUnitSearchRequest(
                name_pattern=name_pattern or "",
                frame_pattern=frame_pattern or "",
                max_results=max_results
            )
            response = self.stub.LexicalUnitSearch(request)
            
            lexical_units = []
            for lu_info in response.lexical_units:
                lexical_units.append({
                    'id': lu_info.id,
                    'name': lu_info.name,
                    'pos': lu_info.pos,
                    'definition': lu_info.definition,
                    'frame_name': lu_info.frame_name,
                    'frame_id': lu_info.frame_id
                })
            
            return {
                'lexical_units': lexical_units,
                'total_count': response.total_count
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC error in search_lexical_units: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in search_lexical_units: {e}")
            raise

    def get_frame_relations(self, frame_id: int = None, frame_name: str = None, 
                          relation_type: str = None) -> Dict[str, Any]:
        """Get frame relations."""
        try:
            if frame_id:
                request = nlp_pb2.FrameRelationsRequest(frame_id=frame_id, relation_type=relation_type or "")
            elif frame_name:
                request = nlp_pb2.FrameRelationsRequest(frame_name=frame_name, relation_type=relation_type or "")
            else:
                raise ValueError("Either frame_id or frame_name must be provided")
            
            response = self.stub.FrameRelations(request)
            
            relations = []
            for rel in response.relations:
                relations.append({
                    'id': rel.id,
                    'type': rel.type,
                    'parent_frame': rel.parent_frame,
                    'child_frame': rel.child_frame,
                    'description': rel.description
                })
            
            return {
                'relations': relations,
                'relation_types': list(response.relation_types)
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC error in get_frame_relations: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in get_frame_relations: {e}")
            raise

    def semantic_role_labeling(self, text: str, include_frame_elements: bool = True) -> List[Dict[str, Any]]:
        """Extract semantic roles from text using FrameNet."""
        try:
            request = nlp_pb2.SemanticRoleLabelingRequest(
                text=text,
                include_frame_elements=include_frame_elements
            )
            response = self.stub.SemanticRoleLabeling(request)
            
            semantic_frames = []
            for frame in response.semantic_frames:
                roles = []
                for role in frame.roles:
                    roles.append({
                        'role_name': role.role_name,
                        'role_type': role.role_type,
                        'text': role.text,
                        'start': role.start,
                        'end': role.end
                    })
                
                semantic_frames.append({
                    'frame_name': frame.frame_name,
                    'frame_id': frame.frame_id,
                    'trigger_word': frame.trigger_word,
                    'trigger_start': frame.trigger_start,
                    'trigger_end': frame.trigger_end,
                    'roles': roles
                })
            
            return semantic_frames
        except grpc.RpcError as e:
            logger.error(f"gRPC error in semantic_role_labeling: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in semantic_role_labeling: {e}")
            raise
