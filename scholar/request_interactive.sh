#!/bin/bash
# ============================================================================
# INTERACTIVE GPU JOB FOR TESTING/DEBUGGING
# ============================================================================
# This script requests an interactive session with a GPU for testing
# your code before submitting long batch jobs.
#
# Usage: bash request_interactive.sh
# ============================================================================

echo "Requesting interactive GPU session on Scholar..."
echo ""
echo "This will give you:"
echo "  - 1 GPU node"
echo "  - 8 CPU cores"
echo "  - 32GB RAM"
echo "  - 2 hours of time"
echo ""
echo "Once you get the session, you can:"
echo "  1. Load modules: module load cuda/11.8 anaconda"
echo "  2. Activate env: source activate quantum-ml"
echo "  3. Test your code: python train_quantum_model.py"
echo ""
echo "Requesting session now..."
echo ""

sinteractive \
    --account=gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --gpus-per-node=1 \
    --time=02:00:00 \
    --mem=32G
