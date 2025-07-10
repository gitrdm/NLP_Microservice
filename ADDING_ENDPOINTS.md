# Adding New Endpoints to the NLP Microservice

This guide explains how to add new endpoints to the gRPC-based NLP microservice.

## Overview

The NLP microservice uses gRPC for communication between client and server. To add a new endpoint, you need to modify 4 main files and follow a specific procedure.

## Current Available Endpoints

| Endpoint | Purpose | Client Method | Parameters |
|----------|---------|---------------|------------|
| `Tokenize` | Break text into tokens | `tokenize()` | `text`, `language` |
| `SentimentAnalysis` | Analyze sentiment | `analyze_sentiment()` | `text` |
| `NamedEntityRecognition` | Extract named entities | `extract_named_entities()` | `text` |
| `POSTagging` | Part-of-speech tagging | `pos_tagging()` | `text` |
| `TextSimilarity` | Calculate text similarity | `calculate_similarity()` | `text1`, `text2` |
| `KeywordExtraction` | Extract keywords | `extract_keywords()` | `text`, `max_keywords` |
| `TextSummarization` | Summarize text | `summarize_text()` | `text`, `max_sentences` |

### FrameNet Endpoints

| Endpoint | Purpose | Client Method | Parameters |
|----------|---------|---------------|------------|
| `FrameSearch` | Search for frames | `search_frames()` | `name_pattern`, `max_results` |
| `FrameDetails` | Get frame details | `get_frame_details()` | `frame_id` or `frame_name` |
| `LexicalUnitSearch` | Search lexical units | `search_lexical_units()` | `name_pattern`, `frame_pattern`, `max_results` |
| `FrameRelations` | Get frame relations | `get_frame_relations()` | `frame_id` or `frame_name`, `relation_type` |
| `SemanticRoleLabeling` | Extract semantic roles | `semantic_role_labeling()` | `text`, `include_frame_elements` |

### WordNet/Synset Endpoints

| Endpoint | Purpose | Client Method | Parameters |
|----------|---------|---------------|------------|
| `SynsetsLookup` | Look up synsets for a word | `lookup_synsets()` | `word`, `pos`, `lang` |
| `SynsetDetails` | Get detailed synset information | `get_synset_details()` | `synset_name`, `include_relations`, `include_examples` |
| `SynsetSimilarity` | Calculate synset similarity | `calculate_synset_similarity()` | `synset1`, `synset2`, `similarity_type` |
| `SynsetRelations` | Get synset relations | `get_synset_relations()` | `synset_name`, `relation_type`, `max_depth` |
| `LemmaSearch` | Search lemmas by pattern | `search_lemmas()` | `lemma_pattern`, `pos`, `lang`, `max_results` |
| `SynonymSearch` | Search synonyms for a word | `search_synonyms()` | `word`, `pos`, `lang`, `include_definitions` |

## Step-by-Step Procedure

### Step 1: Define the Endpoint in the Proto File

**File:** `proto/nlp.proto`

1. Add the RPC service definition inside the `service NLPService` block:
```protobuf
service NLPService {
  // ... existing RPCs ...
  
  // Your new endpoint
  rpc YourNewEndpoint(YourRequest) returns (YourResponse);
}
```

2. Add the request and response message definitions at the end of the file:
```protobuf
message YourRequest {
  string text = 1;
  // Add other fields as needed
  int32 optional_param = 2;
}

message YourResponse {
  string result = 1;
  // Add other fields as needed
  repeated string items = 2;
}
```

### Step 2: Implement the Core Logic

**File:** `src/nlpmicroservice/nlp_service.py`

Add the processing method to the `NLPProcessor` class:

```python
def your_new_method(self, text: str, optional_param: int = 0) -> Dict[str, Any]:
    """Your new NLP processing method."""
    try:
        # Implement your NLP logic here
        # Example:
        tokens = word_tokenize(text)
        
        # Process the tokens
        result = "processed_result"
        items = ["item1", "item2"]
        
        return {
            'result': result,
            'items': items
        }
    except Exception as e:
        logger.error(f"Error in your_new_method: {e}")
        raise
```

### Step 3: Add the Server Endpoint Handler

**File:** `src/nlpmicroservice/server.py`

Add the gRPC endpoint handler to the `NLPServicer` class:

```python
def YourNewEndpoint(self, request, context):
    """Handle your new endpoint requests."""
    try:
        logger.info(f"YourNewEndpoint request: text length={len(request.text)}")
        
        # Extract parameters from request
        optional_param = request.optional_param if request.optional_param > 0 else 0
        
        # Call the processing method
        result = self.nlp_processor.your_new_method(request.text, optional_param)
        
        # Return the response
        return nlp_pb2.YourResponse(
            result=result['result'],
            items=result['items']
        )
    except Exception as e:
        logger.error(f"YourNewEndpoint error: {e}")
        context.set_code(grpc.StatusCode.INTERNAL)
        context.set_details(f"Your new endpoint failed: {e}")
        return nlp_pb2.YourResponse()
```

### Step 4: Add the Client Method

**File:** `src/nlpmicroservice/client.py`

Add the client wrapper method to the `NLPClient` class:

```python
def your_new_method(self, text: str, optional_param: int = 0) -> Dict[str, Any]:
    """Call your new endpoint."""
    try:
        if not self.stub:
            raise Exception("Client not connected")
        
        request = nlp_pb2.YourRequest(text=text, optional_param=optional_param)
        response = self.stub.YourNewEndpoint(request)
        
        return {
            'result': response.result,
            'items': list(response.items)
        }
    except grpc.RpcError as e:
        logger.error(f"gRPC error in your_new_method: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in your_new_method: {e}")
        raise
```

### Step 5: Regenerate gRPC Code

Run the following command to generate the gRPC Python code:

```bash
make generate-proto
```

Or manually:
```bash
poetry run python -m grpc_tools.protoc \
    --proto_path=proto \
    --python_out=src/nlpmicroservice \
    --grpc_python_out=src/nlpmicroservice \
    proto/nlp.proto
```

### Step 6: Fix Import Issues (if needed)

The generated `nlp_pb2_grpc.py` file might need import fixes. Change:
```python
import nlp_pb2 as nlp__pb2
```
to:
```python
from . import nlp_pb2 as nlp__pb2
```

### Step 7: Restart the Server

Stop the running server and start it again:
```bash
# Stop the server
pkill -f "nlp-server"

# Start the server
poetry run nlp-server
```

### Step 8: Test Your New Endpoint

Create a test script to verify your new endpoint works:

```python
#!/usr/bin/env python3
"""Test script for your new endpoint."""

from nlpmicroservice.client import NLPClient
from nlpmicroservice.config import settings

def test_your_new_endpoint():
    """Test your new endpoint."""
    
    sample_text = "Your test text here"
    
    try:
        with NLPClient(host="localhost", port=settings.server_port) as client:
            result = client.your_new_method(sample_text, optional_param=5)
            
            print(f"Result: {result['result']}")
            print(f"Items: {result['items']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_your_new_endpoint()
```

## Example: Text Summarization Endpoint

Here's a complete example of how the `TextSummarization` endpoint was added:

### 1. Proto Definition
```protobuf
service NLPService {
  // ... existing RPCs ...
  rpc TextSummarization(SummarizationRequest) returns (SummarizationResponse);
}

message SummarizationRequest {
  string text = 1;
  int32 max_sentences = 2;
}

message SummarizationResponse {
  string summary = 1;
  int32 original_sentence_count = 2;
  int32 summary_sentence_count = 3;
}
```

### 2. Service Implementation
```python
def summarize_text(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
    """Summarize text using extractive summarization."""
    try:
        sentences = sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return {
                'summary': text,
                'original_sentence_count': len(sentences),
                'summary_sentence_count': len(sentences)
            }
        
        # Scoring logic here...
        
        return {
            'summary': summary,
            'original_sentence_count': len(sentences),
            'summary_sentence_count': len(summary_sentences)
        }
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        raise
```

### 3. Server Handler
```python
def TextSummarization(self, request, context):
    """Handle text summarization requests."""
    try:
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
```

### 4. Client Method
```python
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
```

## Common Endpoint Ideas

Here are some common NLP endpoints you might want to add:

- **Language Detection**: Detect the language of input text
- **Text Classification**: Classify text into categories
- **Question Answering**: Answer questions based on context
- **Text Translation**: Translate text between languages
- **Dependency Parsing**: Parse grammatical dependencies
- **Coreference Resolution**: Resolve pronouns and references
- **Text Generation**: Generate text based on prompts
- **Document Clustering**: Group similar documents
- **Spell Checking**: Check and correct spelling
- **Reading Difficulty**: Assess text readability level

## Tips and Best Practices

1. **Error Handling**: Always include proper error handling in all methods
2. **Logging**: Add appropriate logging for debugging
3. **Validation**: Validate input parameters
4. **Documentation**: Document your methods and parameters
5. **Testing**: Create comprehensive tests for new endpoints
6. **Performance**: Consider performance implications of new processing
7. **Dependencies**: Add any new dependencies to `pyproject.toml`

## Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure to fix import statements in generated files
2. **Server Not Starting**: Check logs for detailed error messages
3. **Method Not Found**: Ensure you've regenerated the gRPC code
4. **Client Connection Issues**: Verify the server is running and accessible

### Debugging Steps:

1. Check server logs: `tail -f logs/nlp_server.log`
2. Test with a simple client script
3. Verify proto file syntax
4. Ensure all files are saved before regenerating code

## Files Modified for Each New Endpoint

| File | Purpose | Changes |
|------|---------|---------|
| `proto/nlp.proto` | gRPC service definition | Add RPC definition and messages |
| `src/nlpmicroservice/nlp_service.py` | Core processing logic | Add processing method |
| `src/nlpmicroservice/server.py` | gRPC server handlers | Add endpoint handler |
| `src/nlpmicroservice/client.py` | Client interface | Add client method |
| `src/nlpmicroservice/nlp_pb2_grpc.py` | Generated gRPC code | Auto-generated (fix imports) |
| `src/nlpmicroservice/nlp_pb2.py` | Generated proto messages | Auto-generated |

Remember to regenerate the gRPC code after modifying the proto file, and always test your new endpoints thoroughly!
