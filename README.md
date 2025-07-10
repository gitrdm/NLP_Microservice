# NLP Microservice

A gRPC-based microservice for Natural Language Processing using NLTK toolkit.

## Features

- **Tokenization**: Break text into individual words and tokens
- **Sentiment Analysis**: Analyze emotional tone of text (positive, negative, neutral)
- **Named Entity Recognition**: Extract entities like people, organizations, locations
- **Part-of-Speech Tagging**: Identify grammatical roles of words
- **Text Similarity**: Calculate similarity between two texts
- **Keyword Extraction**: Extract important keywords from text

## Technology Stack

- **Python 3.11**: Core programming language
- **gRPC**: High-performance RPC framework
- **NLTK**: Natural Language Toolkit for NLP operations
- **Conda**: Virtual environment management
- **Poetry**: Dependency management and packaging
- **Protocol Buffers**: Data serialization

## Project Structure

```
nlpms/
├── src/
│   └── nlpmicroservice/
│       ├── __init__.py
│       ├── server.py          # gRPC server implementation
│       ├── client.py          # gRPC client for testing
│       ├── nlp_service.py     # Core NLP functionality
│       └── config.py          # Configuration settings
├── proto/
│   └── nlp.proto             # gRPC service definition
├── tests/
│   └── test_nlp_service.py   # Test suite
├── logs/                     # Application logs
├── environment.yml           # Conda environment
├── pyproject.toml           # Poetry dependencies
├── Makefile                 # Build and development tasks
├── .env                     # Environment variables
└── README.md               # This file
```

## Quick Start

### 1. Set up the environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate nlpmicroservice

# Install dependencies with Poetry
poetry install

# Generate gRPC code from proto files
make generate-proto
```

### 2. Start the server

```bash
# Run the server (listens on port 8161)
poetry run nlp-server

# Or directly with Python
python -m nlpmicroservice.server
```

### 3. Test the client

```bash
# Run example client
python -m nlpmicroservice.client
```

## API Reference

### gRPC Service: `NLPService`

#### Methods

1. **Tokenize**
   - Input: `TokenizeRequest{text: string, language: string}`
   - Output: `TokenizeResponse{tokens: string[], token_count: int32}`

2. **SentimentAnalysis**
   - Input: `SentimentRequest{text: string}`
   - Output: `SentimentResponse{sentiment: string, confidence: double}`

3. **NamedEntityRecognition**
   - Input: `NERRequest{text: string}`
   - Output: `NERResponse{entities: NamedEntity[]}`

4. **POSTagging**
   - Input: `POSRequest{text: string}`
   - Output: `POSResponse{tags: POSTag[]}`

5. **TextSimilarity**
   - Input: `SimilarityRequest{text1: string, text2: string}`
   - Output: `SimilarityResponse{similarity: double}`

6. **KeywordExtraction**
   - Input: `KeywordRequest{text: string, max_keywords: int32}`
   - Output: `KeywordResponse{keywords: Keyword[]}`

## Configuration

Environment variables can be set in `.env` file:

```env
SERVER_HOST=0.0.0.0
SERVER_PORT=8161
MAX_WORKERS=10
LOG_LEVEL=INFO
```

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=nlpmicroservice

# Run integration tests
poetry run pytest -m integration
```

### Code Quality

```bash
# Format code
poetry run black .

# Sort imports
poetry run isort .

# Lint code
poetry run flake8 .

# Type checking
poetry run mypy .
```

### Generate gRPC Code

```bash
# Generate Python code from proto files
make generate-proto
```

## Usage Examples

### Python Client

```python
from nlpmicroservice.client import NLPClient

with NLPClient() as client:
    # Tokenize text
    result = client.tokenize("Hello world!")
    print(f"Tokens: {result['tokens']}")
    
    # Analyze sentiment
    sentiment = client.analyze_sentiment("I love this product!")
    print(f"Sentiment: {sentiment['sentiment']}")
    
    # Extract keywords
    keywords = client.extract_keywords("Natural language processing is amazing")
    print(f"Keywords: {keywords}")
```

### gRPC Client (any language)

```python
import grpc
from nlpmicroservice import nlp_pb2, nlp_pb2_grpc

# Connect to server
channel = grpc.insecure_channel('localhost:8161')
stub = nlp_pb2_grpc.NLPServiceStub(channel)

# Make request
request = nlp_pb2.TokenizeRequest(text="Hello world!")
response = stub.Tokenize(request)
print(f"Tokens: {list(response.tokens)}")
```

## Performance

The service is designed for high-performance text processing:

- **Concurrent Processing**: Uses ThreadPoolExecutor for handling multiple requests
- **Memory Efficient**: Streaming and batch processing support
- **Scalable**: Can be deployed in containerized environments
- **Fast**: Optimized NLTK operations with caching

## Monitoring

Logs are written to:
- Console (with structured logging)
- `logs/nlp_server.log` (with rotation)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `poetry run pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Roadmap

- [ ] Add more NLP operations (summarization, translation)
- [ ] Add authentication and authorization
- [ ] Add metrics and monitoring
- [ ] Add Docker containerization
- [ ] Add distributed processing support
- [ ] Add model fine-tuning capabilities
