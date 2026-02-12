#!/bin/bash
# Script to run all sentiment classification training scripts
# MLP and LSTM run from STAT359 root directory
# RNN, GRU, BERT, GPT run from Assignment_3 directory

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the STAT359 root directory (parent of student directory)
STAT359_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "=========================================="
echo "Running all sentiment classification models"
echo "=========================================="
echo "STAT359 Root: $STAT359_ROOT"
echo "Assignment_3 Dir: $SCRIPT_DIR"
echo ""

# Function to run a training script
run_training() {
    local script_name=$1
    local working_dir=$2
    local script_path=$3
    
    echo ""
    echo "=========================================="
    echo "Training: $script_name"
    echo "Working Directory: $working_dir"
    echo "=========================================="
    
    cd "$working_dir"
    poetry run python "$script_path"
    
    if [ $? -eq 0 ]; then
        echo "✓ $script_name completed successfully"
    else
        echo "✗ $script_name failed"
        exit 1
    fi
}

# Run MLP and LSTM from STAT359 root
echo "Running MLP and LSTM from STAT359 root..."
run_training "MLP" "$STAT359_ROOT" "student/Assignment_3/train_sentiment_mlp_classifier.py"
run_training "LSTM" "$STAT359_ROOT" "student/Assignment_3/train_sentiment_lstm_classifier.py"

# Run RNN, GRU, BERT, GPT from Assignment_3 directory
echo ""
echo "Running RNN, GRU, BERT, GPT from Assignment_3 directory..."
run_training "RNN" "$SCRIPT_DIR" "train_sentiment_rnn_classifier.py"
run_training "GRU" "$SCRIPT_DIR" "train_sentiment_gru_classifier.py"
run_training "BERT" "$SCRIPT_DIR" "train_sentiment_bert_classifier.py"
run_training "GPT" "$SCRIPT_DIR" "train_sentiment_gpt_classifier.py"

echo ""
echo "=========================================="
echo "All training scripts completed!"
echo "=========================================="
echo "Results saved to: student/Assignment_3/outputs/model_performance.csv"
