# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Configure Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-interaction --no-ansi

# Copy application code
COPY src/ ./src/
COPY proto/ ./proto/

# Generate gRPC code
RUN python -m grpc_tools.protoc \
    --proto_path=proto \
    --python_out=src/nlpmicroservice \
    --grpc_python_out=src/nlpmicroservice \
    proto/nlp.proto

# Download NLTK data
RUN python -c "import nltk; \
    nltk.download('punkt', quiet=True); \
    nltk.download('averaged_perceptron_tagger', quiet=True); \
    nltk.download('maxent_ne_chunker', quiet=True); \
    nltk.download('words', quiet=True); \
    nltk.download('vader_lexicon', quiet=True); \
    nltk.download('stopwords', quiet=True); \
    nltk.download('wordnet', quiet=True)"

# Create logs directory
RUN mkdir -p logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8161

# Expose port
EXPOSE 8161

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grpc; \
        channel = grpc.insecure_channel('localhost:8161'); \
        grpc.channel_ready_future(channel).result(timeout=10)"

# Run the server
CMD ["python", "-m", "nlpmicroservice.server"]
