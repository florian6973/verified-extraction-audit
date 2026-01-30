#!/bin/bash
# Run the complete pipeline for name extraction and evaluation
# Usage: ./run_pipeline.sh [--config path/to/config.yaml] [--eval-only]

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
CONFIG_ARG=""
EVAL_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_ARG="--config $2"
            shift 2
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--config path/to/config.yaml] [--eval-only]"
            echo ""
            echo "Options:"
            echo "  --config PATH    Path to config file (default: config.yaml)"
            echo "  --eval-only      Skip data processing, only run evaluation and plots"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config path/to/config.yaml] [--eval-only]"
            exit 1
            ;;
    esac
done

echo "============================================================"
if [ "$EVAL_ONLY" = true ]; then
    echo "Running EVALUATION ONLY (skipping data processing)"
else
    echo "Running complete pipeline"
fi
echo "============================================================"
echo "Script dir: ${SCRIPT_DIR}"
if [ -n "${CONFIG_ARG}" ]; then
    echo "Config: ${CONFIG_ARG}"
else
    echo "Config: default (config.yaml)"
fi
echo "============================================================"

if [ "$EVAL_ONLY" = false ]; then

# Step 1: Check names (match generated names with df_src)
echo ""
echo "============================================================"
echo "Step 1: Running check_names.py"
echo "============================================================"
python "${SCRIPT_DIR}/check_names.py" ${CONFIG_ARG}

# Step 2: Extract names and compute ll for remaining values
echo ""
echo "============================================================"
echo "Step 2: Running ner_ll_remaining.py (extract first two words)"
echo "============================================================"
python "${SCRIPT_DIR}/ner_ll_remaining.py" ${CONFIG_ARG}

# Step 3: Merge all names
echo ""
echo "============================================================"
echo "Step 3: Running merge_all_names.py"
echo "============================================================"
python "${SCRIPT_DIR}/merge_all_names.py" ${CONFIG_ARG}

# Step 4: Compute ll for all names from base and finetuned models
echo ""
echo "============================================================"
echo "Step 4: Running compute_ll_names.py"
echo "============================================================"
python "${SCRIPT_DIR}/compute_ll_names.py" ${CONFIG_ARG}

fi  # End of data processing steps (--eval-only skips these)

# Step 5: Evaluate classifier
echo ""
echo "============================================================"
echo "Step 5: Running evaluate_classifier.py"
echo "============================================================"
python "${SCRIPT_DIR}/evaluate_classifier.py" ${CONFIG_ARG}

# Step 6: Plot distributions
echo ""
echo "============================================================"
echo "Step 6: Running plot_ll_distribution.py"
echo "============================================================"
python "${SCRIPT_DIR}/plot_ll_distribution.py" ${CONFIG_ARG}

# # Step 7: Analyze classifier by LL percentiles
# echo ""
# echo "============================================================"
# echo "Step 7: Running analyze_clf_by_ll.py"
# echo "============================================================"
# python "${SCRIPT_DIR}/analyze_clf_by_ll.py" ${CONFIG_ARG}

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
