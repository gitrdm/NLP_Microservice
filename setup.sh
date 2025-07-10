#!/bin/bash
# Setup script for NLP Microservice

set -e

echo "=== NLP Microservice Setup ==="
echo

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Create conda environment
echo "Creating conda environment..."
conda env create -f environment.yml

echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate nlpmicroservice

# Install dependencies with Poetry
echo "Installing dependencies with Poetry..."
poetry install

# Generate gRPC code
echo "Generating gRPC code..."
make generate-proto

# Download NLTK data
echo "Downloading NLTK data..."
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
print('NLTK data downloaded successfully')
"

echo
echo "=== Setup Complete ==="
echo
echo "To activate the environment, run:"
echo "  conda activate nlpmicroservice"
echo
echo "To start the server, run:"
echo "  poetry run nlp-server"
echo
echo "To test the client, run:"
echo "  python -m nlpmicroservice.client"
echo
