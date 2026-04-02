#!/bin/bash
# Quick setup script for speaker diarization

set -e

echo "Setting up Speaker Diarization System..."
echo "========================================"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create audio directory if it doesn't exist
if [ ! -d "audio" ]; then
    echo "Creating audio directory..."
    mkdir -p audio
    echo "NOTE: Place your audio/sample.wav file in the audio/ directory"
fi

# Verify diarization_output exists
if [ ! -d "diarization_output" ]; then
    mkdir -p diarization_output
    echo "Created diarization_output directory"
fi

echo ""
echo "Setup complete!"
echo "========================================"
echo "Next steps:"
echo "1. Place your 2-minute WAV file at: audio/sample.wav"
echo "2. Run: python speaker_diarization.py"
echo "3. Check output in: diarization_output/"
echo ""
echo "For detailed documentation, see: DIARIZATION_README.md"
