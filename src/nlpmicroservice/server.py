"""gRPC server implementation for NLP microservice."""

import asyncio
import grpc
from concurrent import futures
from typing import Iterator
from loguru import logger

from .config import settings
from .nlp_service import NLPProcessor

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


class NLPServicer(nlp_pb2_grpc.NLPServiceServicer):
    """gRPC servicer for NLP operations."""
    
    def __init__(self):
        """Initialize the NLP servicer."""
        self.nlp_processor = NLPProcessor()
        logger.info("NLP servicer initialized")
    
    def Tokenize(self, request, context):
        """Tokenize text into words."""
        try:
            logger.info(f"Tokenizing text: {request.text[:100]}...")
            tokens, count = self.nlp_processor.tokenize(
                request.text, 
                request.language or "english"
            )
            
            response = nlp_pb2.TokenizeResponse()
            response.tokens.extend(tokens)
            response.token_count = count
            
            logger.info(f"Tokenization completed. Token count: {count}")
            return response
        except Exception as e:
            logger.error(f"Error in Tokenize: {e}")
            context.set_details(f"Tokenization failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return nlp_pb2.TokenizeResponse()
    
    def SentimentAnalysis(self, request, context):
        """Perform sentiment analysis on text."""
        try:
            logger.info(f"Analyzing sentiment for text: {request.text[:100]}...")
            sentiment, confidence = self.nlp_processor.analyze_sentiment(request.text)
            
            response = nlp_pb2.SentimentResponse()
            response.sentiment = sentiment
            response.confidence = confidence
            
            logger.info(f"Sentiment analysis completed. Sentiment: {sentiment}, Confidence: {confidence}")
            return response
        except Exception as e:
            logger.error(f"Error in SentimentAnalysis: {e}")
            context.set_details(f"Sentiment analysis failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return nlp_pb2.SentimentResponse()
    
    def NamedEntityRecognition(self, request, context):
        """Extract named entities from text."""
        try:
            logger.info(f"Extracting named entities from text: {request.text[:100]}...")
            entities = self.nlp_processor.extract_named_entities(request.text)
            
            response = nlp_pb2.NERResponse()
            for entity in entities:
                ne = response.entities.add()
                ne.text = entity['text']
                ne.label = entity['label']
                ne.start = entity['start']
                ne.end = entity['end']
            
            logger.info(f"Named entity recognition completed. Found {len(entities)} entities")
            return response
        except Exception as e:
            logger.error(f"Error in NamedEntityRecognition: {e}")
            context.set_details(f"Named entity recognition failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return nlp_pb2.NERResponse()
    
    def POSTagging(self, request, context):
        """Perform part-of-speech tagging."""
        try:
            logger.info(f"Performing POS tagging for text: {request.text[:100]}...")
            pos_tags = self.nlp_processor.pos_tagging(request.text)
            
            response = nlp_pb2.POSResponse()
            for word, tag in pos_tags:
                pos_tag = response.tags.add()
                pos_tag.word = word
                pos_tag.tag = tag
            
            logger.info(f"POS tagging completed. Tagged {len(pos_tags)} words")
            return response
        except Exception as e:
            logger.error(f"Error in POSTagging: {e}")
            context.set_details(f"POS tagging failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return nlp_pb2.POSResponse()
    
    def TextSimilarity(self, request, context):
        """Calculate text similarity."""
        try:
            logger.info(f"Calculating similarity between texts...")
            similarity = self.nlp_processor.calculate_similarity(request.text1, request.text2)
            
            response = nlp_pb2.SimilarityResponse()
            response.similarity = similarity
            
            logger.info(f"Text similarity calculation completed. Similarity: {similarity}")
            return response
        except Exception as e:
            logger.error(f"Error in TextSimilarity: {e}")
            context.set_details(f"Text similarity calculation failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return nlp_pb2.SimilarityResponse()
    
    def KeywordExtraction(self, request, context):
        """Extract keywords from text."""
        try:
            logger.info(f"Extracting keywords from text: {request.text[:100]}...")
            max_keywords = request.max_keywords or 10
            keywords = self.nlp_processor.extract_keywords(request.text, max_keywords)
            
            response = nlp_pb2.KeywordResponse()
            for word, score in keywords:
                keyword = response.keywords.add()
                keyword.word = word
                keyword.score = score
            
            logger.info(f"Keyword extraction completed. Found {len(keywords)} keywords")
            return response
        except Exception as e:
            logger.error(f"Error in KeywordExtraction: {e}")
            context.set_details(f"Keyword extraction failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return nlp_pb2.KeywordResponse()

    def TextSummarization(self, request, context):
        """Handle text summarization requests."""
        try:
            logger.info(f"TextSummarization request: text length={len(request.text)}, max_sentences={request.max_sentences}")
            
            max_sentences = request.max_sentences if request.max_sentences > 0 else 3
            result = self.nlp_processor.summarize_text(request.text, max_sentences)
            
            return nlp_pb2.SummarizationResponse(
                summary=result['summary'],
                original_sentence_count=result['original_sentence_count'],
                summary_sentence_count=result['summary_sentence_count']
            )
        except Exception as e:
            logger.error(f"TextSummarization error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Text summarization failed: {e}")
            return nlp_pb2.SummarizationResponse()

    def FrameSearch(self, request, context):
        """Handle frame search requests."""
        try:
            logger.info(f"FrameSearch request: pattern={request.name_pattern}, max_results={request.max_results}")
            
            name_pattern = request.name_pattern if request.name_pattern else None
            max_results = request.max_results if request.max_results > 0 else 50
            
            frames = self.nlp_processor.search_frames(name_pattern, max_results)
            
            # Convert to protobuf format
            frame_infos = []
            for frame in frames:
                frame_info = nlp_pb2.FrameInfo(
                    id=frame['id'],
                    name=frame['name'],
                    definition=frame['definition'],
                    lexical_units=frame['lexical_units']
                )
                frame_infos.append(frame_info)
            
            return nlp_pb2.FrameSearchResponse(
                frames=frame_infos,
                total_count=len(frames)
            )
        except Exception as e:
            logger.error(f"FrameSearch error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Frame search failed: {e}")
            return nlp_pb2.FrameSearchResponse()

    def FrameDetails(self, request, context):
        """Handle frame details requests."""
        try:
            logger.info(f"FrameDetails request: id={request.frame_id}, name={request.frame_name}")
            
            frame_id = request.frame_id if request.frame_id > 0 else None
            frame_name = request.frame_name if request.frame_name else None
            
            frame_details = self.nlp_processor.get_frame_details(frame_id, frame_name)
            
            # Convert lexical units
            lexical_units = []
            for lu in frame_details['lexical_units']:
                lexical_units.append(nlp_pb2.LexicalUnit(
                    id=lu['id'],
                    name=lu['name'],
                    pos=lu['pos'],
                    definition=lu['definition'],
                    status=lu['status']
                ))
            
            # Convert frame elements
            frame_elements = []
            for fe in frame_details['frame_elements']:
                frame_elements.append(nlp_pb2.FrameElement(
                    id=fe['id'],
                    name=fe['name'],
                    definition=fe['definition'],
                    core_type=fe['core_type'],
                    abbreviation=fe['abbreviation']
                ))
            
            # Convert frame relations
            frame_relations = []
            for rel in frame_details['frame_relations']:
                frame_relations.append(nlp_pb2.FrameRelation(
                    id=rel['id'],
                    type=rel['type'],
                    parent_frame=rel['parent_frame'],
                    child_frame=rel['child_frame'],
                    description=rel['description']
                ))
            
            # Convert semantic types
            semantic_types = []
            for st in frame_details['semantic_types']:
                semantic_types.append(nlp_pb2.SemanticType(
                    id=st['id'],
                    name=st['name'],
                    abbreviation=st['abbreviation'],
                    definition=st['definition']
                ))
            
            return nlp_pb2.FrameDetailsResponse(
                id=frame_details['id'],
                name=frame_details['name'],
                definition=frame_details['definition'],
                lexical_units=lexical_units,
                frame_elements=frame_elements,
                frame_relations=frame_relations,
                semantic_types=semantic_types
            )
        except Exception as e:
            logger.error(f"FrameDetails error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Frame details failed: {e}")
            return nlp_pb2.FrameDetailsResponse()

    def LexicalUnitSearch(self, request, context):
        """Handle lexical unit search requests."""
        try:
            logger.info(f"LexicalUnitSearch request: name_pattern={request.name_pattern}")
            
            name_pattern = request.name_pattern if request.name_pattern else None
            frame_pattern = request.frame_pattern if request.frame_pattern else None
            max_results = request.max_results if request.max_results > 0 else 50
            
            lus = self.nlp_processor.search_lexical_units(name_pattern, frame_pattern, max_results)
            
            # Convert to protobuf format
            lu_infos = []
            for lu in lus:
                lu_info = nlp_pb2.LexicalUnitInfo(
                    id=lu['id'],
                    name=lu['name'],
                    pos=lu['pos'],
                    definition=lu['definition'],
                    frame_name=lu['frame_name'],
                    frame_id=lu['frame_id']
                )
                lu_infos.append(lu_info)
            
            return nlp_pb2.LexicalUnitSearchResponse(
                lexical_units=lu_infos,
                total_count=len(lus)
            )
        except Exception as e:
            logger.error(f"LexicalUnitSearch error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Lexical unit search failed: {e}")
            return nlp_pb2.LexicalUnitSearchResponse()

    def FrameRelations(self, request, context):
        """Handle frame relations requests."""
        try:
            logger.info(f"FrameRelations request: id={request.frame_id}, name={request.frame_name}")
            
            frame_id = request.frame_id if request.frame_id > 0 else None
            frame_name = request.frame_name if request.frame_name else None
            relation_type = request.relation_type if request.relation_type else None
            
            relations_data = self.nlp_processor.get_frame_relations(frame_id, frame_name, relation_type)
            
            # Convert relations
            relations = []
            for rel in relations_data['relations']:
                relations.append(nlp_pb2.FrameRelation(
                    id=rel['id'],
                    type=rel['type'],
                    parent_frame=rel['parent_frame'],
                    child_frame=rel['child_frame'],
                    description=rel['description']
                ))
            
            return nlp_pb2.FrameRelationsResponse(
                relations=relations,
                relation_types=relations_data['relation_types']
            )
        except Exception as e:
            logger.error(f"FrameRelations error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Frame relations failed: {e}")
            return nlp_pb2.FrameRelationsResponse()

    def SemanticRoleLabeling(self, request, context):
        """Handle semantic role labeling requests."""
        try:
            logger.info(f"SemanticRoleLabeling request: text length={len(request.text)}")
            
            semantic_frames = self.nlp_processor.semantic_role_labeling(
                request.text, 
                request.include_frame_elements
            )
            
            # Convert to protobuf format
            pb_frames = []
            for frame in semantic_frames:
                # Convert roles
                roles = []
                for role in frame['roles']:
                    roles.append(nlp_pb2.SemanticRole(
                        role_name=role['role_name'],
                        role_type=role['role_type'],
                        text=role['text'],
                        start=role['start'],
                        end=role['end']
                    ))
                
                pb_frame = nlp_pb2.SemanticFrame(
                    frame_name=frame['frame_name'],
                    frame_id=frame['frame_id'],
                    trigger_word=frame['trigger_word'],
                    trigger_start=frame['trigger_start'],
                    trigger_end=frame['trigger_end'],
                    roles=roles
                )
                pb_frames.append(pb_frame)
            
            return nlp_pb2.SemanticRoleLabelingResponse(
                semantic_frames=pb_frames
            )
        except Exception as e:
            logger.error(f"SemanticRoleLabeling error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Semantic role labeling failed: {e}")
            return nlp_pb2.SemanticRoleLabelingResponse()


def serve():
    """Start the gRPC server."""
    logger.info("Starting NLP gRPC server...")
    
    # Create server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=settings.max_workers))
    
    # Add servicer
    nlp_pb2_grpc.add_NLPServiceServicer_to_server(NLPServicer(), server)
    
    # Add port
    listen_addr = f"{settings.server_host}:{settings.server_port}"
    server.add_insecure_port(listen_addr)
    
    # Start server
    server.start()
    logger.info(f"NLP gRPC server started on {listen_addr}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)


def main():
    """Main entry point."""
    logger.info("NLP Microservice starting up...")
    logger.info(f"Configuration: {settings.model_dump()}")
    serve()


if __name__ == "__main__":
    main()
