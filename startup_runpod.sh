#!/bin/bash

cd /workspace/alignment-lab || exit 1

echo "Pulling latest code..."
git pull

echo "Installing dependencies..."
python -m pip install -r requirements.txt

# HuggingFace login (only if not already logged in)
if ! hf whoami &> /dev/null; then
    echo "Logging into HuggingFace..."
    hf auth login
else
    echo "Already logged into HuggingFace"
fi

# Set Anthropic API key (only if not already set)
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Enter your Anthropic API key:"
    read -s ANTHROPIC_API_KEY
    export ANTHROPIC_API_KEY
else
    echo "Anthropic API key already set"
fi

echo "Startup complete!"