"""NLP service implementation using NLTK."""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from typing import List, Tuple, Dict, Any
import math
import re
from loguru import logger

from .config import settings


class NLPProcessor:
    """Natural Language Processing service using NLTK."""
    
    def __init__(self):
        """Initialize the NLP processor."""
        self._download_nltk_data()
        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def _download_nltk_data(self) -> None:
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('chunkers/maxent_ne_chunker')
            nltk.data.find('corpora/words')
            nltk.data.find('vader_lexicon')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            logger.info("NLTK data already available")
        except LookupError:
            logger.info("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            logger.info("NLTK data downloaded successfully")
    
    def tokenize(self, text: str, language: str = "english") -> Tuple[List[str], int]:
        """Tokenize text into words."""
        try:
            tokens = word_tokenize(text, language=language)
            # Remove punctuation and convert to lowercase
            tokens = [token.lower() for token in tokens if token.isalnum()]
            return tokens, len(tokens)
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment of text."""
        try:
            scores = self.sia.polarity_scores(text)
            compound_score = scores['compound']
            
            if compound_score >= 0.05:
                sentiment = "positive"
            elif compound_score <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            confidence = abs(compound_score)
            return sentiment, confidence
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            raise
    
    def extract_named_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)
            
            entities = []
            current_entity = []
            current_label = None
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    # This is a named entity
                    if current_label != chunk.label():
                        # New entity type
                        if current_entity:
                            entity_text = ' '.join([word for word, _ in current_entity])
                            entities.append({
                                'text': entity_text,
                                'label': current_label,
                                'start': 0,  # Simplified for now
                                'end': 0     # Simplified for now
                            })
                        current_entity = list(chunk)
                        current_label = chunk.label()
                    else:
                        # Continue current entity
                        current_entity.extend(list(chunk))
                else:
                    # Not a named entity
                    if current_entity:
                        entity_text = ' '.join([word for word, _ in current_entity])
                        entities.append({
                            'text': entity_text,
                            'label': current_label,
                            'start': 0,  # Simplified for now
                            'end': 0     # Simplified for now
                        })
                        current_entity = []
                        current_label = None
            
            # Handle last entity
            if current_entity:
                entity_text = ' '.join([word for word, _ in current_entity])
                entities.append({
                    'text': entity_text,
                    'label': current_label,
                    'start': 0,  # Simplified for now
                    'end': 0     # Simplified for now
                })
            
            return entities
        except Exception as e:
            logger.error(f"Error extracting named entities: {e}")
            raise
    
    def pos_tagging(self, text: str) -> List[Tuple[str, str]]:
        """Perform part-of-speech tagging."""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            return pos_tags
        except Exception as e:
            logger.error(f"Error performing POS tagging: {e}")
            raise
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using cosine similarity."""
        try:
            # Tokenize and process texts
            tokens1 = self.tokenize(text1)[0]
            tokens2 = self.tokenize(text2)[0]
            
            # Remove stop words and lemmatize
            processed1 = [self.lemmatizer.lemmatize(token) for token in tokens1 
                         if token not in self.stop_words]
            processed2 = [self.lemmatizer.lemmatize(token) for token in tokens2 
                         if token not in self.stop_words]
            
            # Create frequency vectors
            all_words = set(processed1 + processed2)
            vector1 = [processed1.count(word) for word in all_words]
            vector2 = [processed2.count(word) for word in all_words]
            
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(vector1, vector2))
            magnitude1 = math.sqrt(sum(a * a for a in vector1))
            magnitude2 = math.sqrt(sum(b * b for b in vector2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            similarity = dot_product / (magnitude1 * magnitude2)
            return similarity
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            raise
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords from text using TF-IDF approach."""
        try:
            # Tokenize and clean text
            tokens = self.tokenize(text)[0]
            
            # Remove stop words and lemmatize
            processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                              if token not in self.stop_words and len(token) > 2]
            
            # Calculate term frequency
            term_freq = Counter(processed_tokens)
            total_terms = len(processed_tokens)
            
            # Simple TF score (normalized frequency)
            keywords = []
            for term, freq in term_freq.most_common(max_keywords):
                tf_score = freq / total_terms
                keywords.append((term, tf_score))
            
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            raise

    def summarize_text(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
        """Summarize text using extractive summarization."""
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            
            if len(sentences) <= max_sentences:
                return {
                    'summary': text,
                    'original_sentence_count': len(sentences),
                    'summary_sentence_count': len(sentences)
                }
            
            # Simple extractive summarization based on sentence scoring
            sentence_scores = {}
            
            for i, sentence in enumerate(sentences):
                # Get keywords for this sentence
                keywords = self.extract_keywords(sentence, max_keywords=5)
                
                # Score based on keyword density and position
                score = sum(kw[1] for kw in keywords)  # Sum of keyword scores
                
                # Boost score for sentences at the beginning
                if i < len(sentences) * 0.3:
                    score *= 1.2
                
                sentence_scores[i] = score
            
            # Select top sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
            
            # Sort by original order
            top_sentences.sort(key=lambda x: x[0])
            
            # Create summary
            summary_sentences = [sentences[i] for i, _ in top_sentences]
            summary = ' '.join(summary_sentences)
            
            return {
                'summary': summary,
                'original_sentence_count': len(sentences),
                'summary_sentence_count': len(summary_sentences)
            }
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            raise

    def search_frames(self, name_pattern: str = None, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for frames by name pattern."""
        try:
            from nltk.corpus import framenet as fn
            
            # Get all frames or filter by pattern
            frames = fn.frames(name_pattern) if name_pattern else fn.frames()
            
            # Limit results
            if max_results and len(frames) > max_results:
                frames = frames[:max_results]
            
            results = []
            for frame in frames:
                # Get basic lexical unit names
                lu_names = [lu.name for lu in frame.lexUnit.values()][:10]  # Limit to first 10
                
                results.append({
                    'id': frame.ID,
                    'name': frame.name,
                    'definition': frame.definition,
                    'lexical_units': lu_names
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching frames: {e}")
            raise

    def get_frame_details(self, frame_id: int = None, frame_name: str = None) -> Dict[str, Any]:
        """Get detailed information about a specific frame."""
        try:
            from nltk.corpus import framenet as fn
            
            # Get frame by ID or name
            if frame_id:
                frame = fn.frame(frame_id)
            elif frame_name:
                frame = fn.frame(frame_name)
            else:
                raise ValueError("Either frame_id or frame_name must be provided")
            
            # Extract lexical units
            lexical_units = []
            for lu in frame.lexUnit.values():
                lexical_units.append({
                    'id': lu.ID,
                    'name': lu.name,
                    'pos': lu.POS,
                    'definition': lu.definition,
                    'status': lu.status
                })
            
            # Extract frame elements
            frame_elements = []
            for fe in frame.FE.values():
                frame_elements.append({
                    'id': fe.ID,
                    'name': fe.name,
                    'definition': fe.definition,
                    'core_type': fe.coreType,
                    'abbreviation': fe.abbrev
                })
            
            # Extract frame relations
            frame_relations = []
            for rel in frame.frameRelations:
                try:
                    # Safely access relation attributes
                    rel_id = getattr(rel, 'ID', 0)
                    rel_type = getattr(rel, 'type', None)
                    rel_type_name = getattr(rel_type, 'name', str(rel_type)) if rel_type else 'Unknown'
                    
                    # Check for parent/child frames
                    parent_frame = ''
                    child_frame = ''
                    
                    if hasattr(rel, 'Parent') and rel.Parent:
                        parent_frame = getattr(rel.Parent, 'name', str(rel.Parent))
                    if hasattr(rel, 'Child') and rel.Child:
                        child_frame = getattr(rel.Child, 'name', str(rel.Child))
                    
                    frame_relations.append({
                        'id': rel_id,
                        'type': rel_type_name,
                        'parent_frame': parent_frame,
                        'child_frame': child_frame,
                        'description': f"{rel_type_name} relationship"
                    })
                except Exception as rel_error:
                    logger.warning(f"Error processing frame relation: {rel_error}")
                    continue
            
            # Extract semantic types
            semantic_types = []
            for st in frame.semTypes:
                semantic_types.append({
                    'id': st.ID,
                    'name': st.name,
                    'abbreviation': st.abbrev if hasattr(st, 'abbrev') else '',
                    'definition': st.definition if hasattr(st, 'definition') else ''
                })
            
            return {
                'id': frame.ID,
                'name': frame.name,
                'definition': frame.definition,
                'lexical_units': lexical_units,
                'frame_elements': frame_elements,
                'frame_relations': frame_relations,
                'semantic_types': semantic_types
            }
        except Exception as e:
            logger.error(f"Error getting frame details: {e}")
            raise

    def search_lexical_units(self, name_pattern: str = None, frame_pattern: str = None, 
                           max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for lexical units."""
        try:
            from nltk.corpus import framenet as fn
            
            # Get lexical units
            lus = fn.lus(name_pattern, frame_pattern)
            
            # Limit results
            if max_results and len(lus) > max_results:
                lus = lus[:max_results]
            
            results = []
            for lu in lus:
                results.append({
                    'id': lu.ID,
                    'name': lu.name,
                    'pos': lu.POS,
                    'definition': lu.definition,
                    'frame_name': lu.frame.name,
                    'frame_id': lu.frame.ID
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching lexical units: {e}")
            raise

    def get_frame_relations(self, frame_id: int = None, frame_name: str = None, 
                          relation_type: str = None) -> Dict[str, Any]:
        """Get frame relations."""
        try:
            from nltk.corpus import framenet as fn
            
            # Get frame identifier
            frame = None
            if frame_id:
                frame = frame_id
            elif frame_name:
                frame = frame_name
            
            # Get relations
            relations = fn.frame_relations(frame=frame, type=relation_type)
            
            # Get available relation types
            relation_types = [rt.name for rt in fn.frame_relation_types()]
            
            # Format relations
            formatted_relations = []
            for rel in relations:
                try:
                    # Safely access relation attributes
                    rel_id = getattr(rel, 'ID', 0)
                    rel_type = getattr(rel, 'type', None)
                    rel_type_name = getattr(rel_type, 'name', str(rel_type)) if rel_type else 'Unknown'
                    
                    # Check for parent/child frames
                    parent_frame = ''
                    child_frame = ''
                    
                    if hasattr(rel, 'Parent') and rel.Parent:
                        parent_frame = getattr(rel.Parent, 'name', str(rel.Parent))
                    if hasattr(rel, 'Child') and rel.Child:
                        child_frame = getattr(rel.Child, 'name', str(rel.Child))
                    
                    formatted_relations.append({
                        'id': rel_id,
                        'type': rel_type_name,
                        'parent_frame': parent_frame,
                        'child_frame': child_frame,
                        'description': f"{rel_type_name} relationship"
                    })
                except Exception as rel_error:
                    logger.warning(f"Error processing frame relation: {rel_error}")
                    continue
            
            return {
                'relations': formatted_relations,
                'relation_types': relation_types
            }
        except Exception as e:
            logger.error(f"Error getting frame relations: {e}")
            raise

    def semantic_role_labeling(self, text: str, include_frame_elements: bool = True) -> List[Dict[str, Any]]:
        """Extract semantic roles from text using FrameNet."""
        try:
            from nltk.corpus import framenet as fn
            
            # This is a simplified implementation
            # In a real implementation, you'd use a proper SRL system
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            semantic_frames = []
            
            # Look for words that might be frame-evoking
            for i, (word, pos) in enumerate(pos_tags):
                word_lower = word.lower()
                
                # Try to find lexical units that match this word
                try:
                    # Search for LUs with this lemma
                    matching_lus = fn.lus(f"^{word_lower}\\.")
                    
                    for lu in matching_lus[:1]:  # Take first match
                        frame = lu.frame
                        
                        # Create semantic frame
                        semantic_frame = {
                            'frame_name': frame.name,
                            'frame_id': frame.ID,
                            'trigger_word': word,
                            'trigger_start': i,
                            'trigger_end': i + 1,
                            'roles': []
                        }
                        
                        # This is a very simplified role assignment
                        # In practice, you'd need a proper SRL system
                        if include_frame_elements:
                            # Add some basic role assignments based on POS patterns
                            roles = []
                            
                            # Look for potential arguments around the trigger
                            for j in range(max(0, i-3), min(len(tokens), i+4)):
                                if j != i:  # Skip the trigger word itself
                                    _, arg_pos = pos_tags[j]
                                    if arg_pos.startswith('NN'):  # Noun phrases
                                        roles.append({
                                            'role_name': 'Participant',  # Generic role
                                            'role_type': 'Core',
                                            'text': tokens[j],
                                            'start': j,
                                            'end': j + 1
                                        })
                            
                            semantic_frame['roles'] = roles[:3]  # Limit to 3 roles
                        
                        semantic_frames.append(semantic_frame)
                        break  # Only one frame per word for simplicity
                        
                except Exception:
                    continue  # Skip words that don't match any LUs
            
            return semantic_frames
        except Exception as e:
            logger.error(f"Error in semantic role labeling: {e}")
            raise

    # WordNet/Synset Processing Methods
    
    def lookup_synsets(self, word: str, pos: str = None, lang: str = 'eng') -> List[Dict[str, Any]]:
        """Look up synsets for a word."""
        try:
            from nltk.corpus import wordnet as wn
            
            # Map POS strings to WordNet constants
            pos_map = {
                'NOUN': wn.NOUN,
                'VERB': wn.VERB, 
                'ADJ': wn.ADJ,
                'ADV': wn.ADV
            }
            
            # Get synsets
            if pos and pos in pos_map:
                synsets = wn.synsets(word, pos=pos_map[pos], lang=lang)
            else:
                synsets = wn.synsets(word, lang=lang)
            
            result = []
            for synset in synsets:
                # Get lemma names in the specified language
                lemma_names = synset.lemma_names(lang) if lang != 'eng' else [lemma.name() for lemma in synset.lemmas()]
                
                synset_info = {
                    'id': synset.name(),
                    'name': synset.name().split('.')[0],
                    'pos': synset.pos().upper(),
                    'definition': synset.definition(),
                    'examples': synset.examples(),
                    'lemma_names': lemma_names,
                    'max_depth': synset.max_depth()
                }
                result.append(synset_info)
                
            return result
        except Exception as e:
            logger.error(f"Error looking up synsets: {e}")
            raise

    def get_synset_details(self, synset_id: str, lang: str = 'eng', 
                          include_relations: bool = True, include_lemmas: bool = True) -> Dict[str, Any]:
        """Get detailed information about a specific synset."""
        try:
            from nltk.corpus import wordnet as wn
            
            synset = wn.synset(synset_id)
            
            # Basic synset info
            lemma_names = synset.lemma_names(lang) if lang != 'eng' else [lemma.name() for lemma in synset.lemmas()]
            
            result = {
                'synset': {
                    'id': synset.name(),
                    'name': synset.name().split('.')[0],
                    'pos': synset.pos().upper(),
                    'definition': synset.definition(),
                    'examples': synset.examples(),
                    'lemma_names': lemma_names,
                    'max_depth': synset.max_depth()
                }
            }
            
            if include_relations:
                # Helper function to convert synsets to info dicts
                def synsets_to_info(synsets):
                    return [{
                        'id': s.name(),
                        'name': s.name().split('.')[0],
                        'pos': s.pos().upper(),
                        'definition': s.definition(),
                        'examples': s.examples()[:2],  # Limit examples
                        'lemma_names': s.lemma_names(lang)[:5] if lang != 'eng' else [lemma.name() for lemma in s.lemmas()][:5],
                        'max_depth': s.max_depth()
                    } for s in synsets]
                
                # Get various relations
                result['hypernyms'] = synsets_to_info(synset.hypernyms())
                result['hyponyms'] = synsets_to_info(synset.hyponyms()[:10])  # Limit hyponyms
                result['meronyms'] = synsets_to_info(synset.part_meronyms() + synset.substance_meronyms() + synset.member_meronyms())
                result['holonyms'] = synsets_to_info(synset.part_holonyms() + synset.substance_holonyms() + synset.member_holonyms())
                result['similar_tos'] = synsets_to_info(synset.similar_tos())
                result['also_sees'] = synsets_to_info(synset.also_sees())
                result['root_hypernyms'] = synsets_to_info(synset.root_hypernyms())
            
            if include_lemmas:
                lemmas = []
                for lemma in synset.lemmas(lang):
                    lemma_info = {
                        'name': lemma.name(),
                        'key': lemma.key(),
                        'count': lemma.count(),
                        'lang': lang,
                        'antonyms': [ant.name() for ant in lemma.antonyms()],
                        'derivationally_related_forms': [drf.name() for drf in lemma.derivationally_related_forms()],
                        'pertainyms': [p.name() for p in lemma.pertainyms()]
                    }
                    lemmas.append(lemma_info)
                result['lemmas'] = lemmas
            
            return result
        except Exception as e:
            logger.error(f"Error getting synset details: {e}")
            raise

    def calculate_synset_similarity(self, synset1_id: str, synset2_id: str, 
                                  similarity_type: str = 'path', simulate_root: bool = True) -> Dict[str, Any]:
        """Calculate similarity between two synsets."""
        try:
            from nltk.corpus import wordnet as wn
            from nltk.corpus import wordnet_ic
            
            synset1 = wn.synset(synset1_id)
            synset2 = wn.synset(synset2_id)
            
            similarity_score = None
            
            # Calculate similarity based on type
            if similarity_type == 'path':
                similarity_score = synset1.path_similarity(synset2, simulate_root=simulate_root)
            elif similarity_type == 'lch':
                similarity_score = synset1.lch_similarity(synset2, simulate_root=simulate_root)
            elif similarity_type == 'wup':
                similarity_score = synset1.wup_similarity(synset2, simulate_root=simulate_root)
            elif similarity_type in ['res', 'jcn', 'lin']:
                # These require information content
                try:
                    brown_ic = wordnet_ic.ic('ic-brown.dat')
                    if similarity_type == 'res':
                        similarity_score = synset1.res_similarity(synset2, brown_ic)
                    elif similarity_type == 'jcn':
                        similarity_score = synset1.jcn_similarity(synset2, brown_ic)
                    elif similarity_type == 'lin':
                        similarity_score = synset1.lin_similarity(synset2, brown_ic)
                except Exception:
                    # Fall back to path similarity if IC data not available
                    similarity_score = synset1.path_similarity(synset2, simulate_root=simulate_root)
            
            # Get lowest common hypernyms
            common_hypernyms = []
            try:
                lch_synsets = synset1.lowest_common_hypernyms(synset2)
                for lch in lch_synsets:
                    common_hypernyms.append({
                        'id': lch.name(),
                        'name': lch.name().split('.')[0],
                        'pos': lch.pos().upper(),
                        'definition': lch.definition(),
                        'examples': lch.examples()[:1],
                        'lemma_names': [lemma.name() for lemma in lch.lemmas()],
                        'max_depth': lch.max_depth()
                    })
            except Exception:
                pass
            
            return {
                'similarity_score': similarity_score if similarity_score is not None else 0.0,
                'similarity_type': similarity_type,
                'synset1_id': synset1_id,
                'synset2_id': synset2_id,
                'common_hypernyms': common_hypernyms
            }
        except Exception as e:
            logger.error(f"Error calculating synset similarity: {e}")
            raise

    def get_synset_relations(self, synset_id: str, relation_type: str = 'hypernyms', 
                           max_depth: int = 3) -> Dict[str, Any]:
        """Get synset relations of a specific type."""
        try:
            from nltk.corpus import wordnet as wn
            
            synset = wn.synset(synset_id)
            
            # Define relation functions
            relation_map = {
                'hypernyms': synset.hypernyms,
                'hyponyms': synset.hyponyms,
                'meronyms': lambda: synset.part_meronyms() + synset.substance_meronyms() + synset.member_meronyms(),
                'holonyms': lambda: synset.part_holonyms() + synset.substance_holonyms() + synset.member_holonyms(),
                'similar_tos': synset.similar_tos,
                'also_sees': synset.also_sees,
                'root_hypernyms': synset.root_hypernyms
            }
            
            if relation_type not in relation_map:
                raise ValueError(f"Unknown relation type: {relation_type}")
            
            # Get direct relations
            related_synsets = relation_map[relation_type]()
            
            # Convert to info format
            synsets_info = []
            for rel_synset in related_synsets[:20]:  # Limit results
                synsets_info.append({
                    'id': rel_synset.name(),
                    'name': rel_synset.name().split('.')[0],
                    'pos': rel_synset.pos().upper(),
                    'definition': rel_synset.definition(),
                    'examples': rel_synset.examples()[:1],
                    'lemma_names': [lemma.name() for lemma in rel_synset.lemmas()][:5],
                    'max_depth': rel_synset.max_depth()
                })
            
            # Get relation paths for closure operations if needed
            relation_paths = []
            if relation_type in ['hypernyms', 'hyponyms'] and max_depth > 1:
                try:
                    if relation_type == 'hypernyms':
                        closure_synsets = synset.closure(synset.hypernyms, depth=max_depth)
                    else:
                        closure_synsets = synset.closure(synset.hyponyms, depth=max_depth)
                    
                    # Create simplified paths (just depth info)
                    for i, rel_synset in enumerate(closure_synsets[:10]):
                        path_info = {
                            'path': [{
                                'id': rel_synset.name(),
                                'name': rel_synset.name().split('.')[0],
                                'pos': rel_synset.pos().upper(),
                                'definition': rel_synset.definition()[:100] + '...' if len(rel_synset.definition()) > 100 else rel_synset.definition(),
                                'examples': [],
                                'lemma_names': [lemma.name() for lemma in rel_synset.lemmas()][:3],
                                'max_depth': rel_synset.max_depth()
                            }],
                            'depth': 1  # Simplified - would need actual path computation
                        }
                        relation_paths.append(path_info)
                except Exception:
                    pass  # Skip closure if it fails
            
            return {
                'synset_id': synset_id,
                'relation_type': relation_type,
                'related_synsets': synsets_info,
                'relation_paths': relation_paths
            }
        except Exception as e:
            logger.error(f"Error getting synset relations: {e}")
            raise

    def search_lemmas(self, lemma_name: str, pos: str = None, lang: str = 'eng', 
                     include_morphology: bool = True) -> Dict[str, Any]:
        """Search for lemmas by name."""
        try:
            from nltk.corpus import wordnet as wn
            
            # Map POS strings to WordNet constants
            pos_map = {
                'NOUN': wn.NOUN,
                'VERB': wn.VERB,
                'ADJ': wn.ADJ,
                'ADV': wn.ADV
            }
            
            # Try morphological analysis first
            morphed_word = lemma_name
            if include_morphology:
                try:
                    if pos and pos in pos_map:
                        morphed = wn.morphy(lemma_name, pos_map[pos])
                    else:
                        morphed = wn.morphy(lemma_name)
                    if morphed:
                        morphed_word = morphed
                except Exception:
                    pass
            
            # Get lemmas
            if pos and pos in pos_map:
                lemmas = wn.lemmas(morphed_word, pos=pos_map[pos], lang=lang)
            else:
                lemmas = wn.lemmas(morphed_word, lang=lang)
            
            lemmas_info = []
            for lemma in lemmas:
                lemma_info = {
                    'name': lemma.name(),
                    'key': lemma.key(),
                    'count': lemma.count(),
                    'lang': lang,
                    'antonyms': [ant.name() for ant in lemma.antonyms()],
                    'derivationally_related_forms': [drf.name() for drf in lemma.derivationally_related_forms()],
                    'pertainyms': [p.name() for p in lemma.pertainyms()]
                }
                lemmas_info.append(lemma_info)
            
            return {
                'lemmas': lemmas_info,
                'original_word': lemma_name,
                'morphed_word': morphed_word,
                'pos': pos or '',
                'lang': lang
            }
        except Exception as e:
            logger.error(f"Error searching lemmas: {e}")
            raise

    def search_synonyms(self, word: str, lang: str = 'eng', max_synonyms: int = 20) -> Dict[str, Any]:
        """Search for synonyms of a word."""
        try:
            from nltk.corpus import wordnet as wn
            
            # Get synonym groups from WordNet
            synonym_groups = []
            
            # Get all synsets for the word
            synsets = wn.synsets(word, lang=lang)
            
            for synset in synsets:
                # Get lemma names (synonyms) for this sense
                if lang != 'eng':
                    synonyms = synset.lemma_names(lang)
                else:
                    synonyms = [lemma.name() for lemma in synset.lemmas()]
                
                # Remove the original word and limit results
                synonyms = [syn for syn in synonyms if syn.lower() != word.lower()][:max_synonyms]
                
                if synonyms:  # Only include if we have synonyms
                    synonym_group = {
                        'sense_definition': synset.definition(),
                        'synonyms': synonyms,
                        'synset_id': synset.name()
                    }
                    synonym_groups.append(synonym_group)
            
            return {
                'word': word,
                'lang': lang,
                'synonym_groups': synonym_groups
            }
        except Exception as e:
            logger.error(f"Error searching synonyms: {e}")
            raise
