#!/bin/bash
# Simple script to ensure python finds the src module
export PYTHONPATH=$(pwd)/src

echo "=== Starting 5B Token Training Loop ==="
uv run python src/train.py
