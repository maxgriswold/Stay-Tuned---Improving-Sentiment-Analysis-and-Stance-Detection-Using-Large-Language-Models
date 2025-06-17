#!/bin/bash

# Analysis Piepline
# Usage: ./run_analysis.sh [--run-gpt]

RUN_GPT=false

initial_time=$(date +%s)

# Activate conda environment:
source activate stay_tuned

# Parse command line arguments. If there is more than one, check
# to see if 
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-gpt)
            RUN_GPT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--run-gpt]"
            exit 1
            ;;
    esac
done

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory to run all commands
cd "$SCRIPT_DIR"

# Create log directory if it doesn't exist
mkdir -p log

# Run commands in both R and Python piping errors and output to log files. 
run_command() {
    local cmd="$1"
    local description="$2"
    local log_file="$3"
    
    echo "Running: $description"
    if [ -n "$log_file" ]; then
        echo "Logging to: $log_file"
    fi
    echo "----------------------------------------"
    
    if [ -n "$log_file" ]; then
        # Run with logging
        if eval "$cmd" > "$log_file" 2>&1; then
			current_time=$(date +%s)
			elapsed_time=$((current_time - initial_time)/60)
            echo "✓ Completed: $description. Took $elapsed_time mins."
            echo ""
        else
            echo "✗ Failed: $description"
            echo "Check log file: $log_file"
            echo "Exiting analysis pipeline due to error."
            exit 1
        fi
    else
        # Run without logging (for bash scripts)
        if eval "$cmd"; then
            echo "✓ Completed: $description"
            echo ""
        else
            echo "✗ Failed: $description"
            echo "Exiting analysis pipeline due to error."
            exit 1
        fi
    fi
}

echo "Starting analysis pipeline..."
echo "Run GPT models: $(if [ "$RUN_GPT" = false ]; then echo "NO"; else echo "YES"; fi)"
echo "========================================"

# Run files in sequence
#run_command "Rscript --no-save --no-restore --verbose code/00_get_analysis_data.R" "Step 00: Get analysis data from Harvard Dataverse" "log/00_get_analysis_data.Rout"

#run_command "Rscript --no-save --no-restore --verbose code/01_prep_analysis_data.R" "Step 01: Prep Twitter data collected by study Team" "log/01_prep_analysis_data.Rout"

#run_command "Rscript --no-save --no-restore --verbose code/02_process_external_datasets.R" "Step 02: Prep Twitter Data from External Studies" "log/02_process_external_datasets.Rout"

#run_command "Rscript --no-save --no-restore --verbose code/03_prep_train_validate_datasets.R" "Step 03: Prepare Training and Validation datasets" "log/03_prep_train_validate_datasets.Rout"

#run_command "Rscript --no-save --no-restore --verbose code/04_lexical_methods.R" "Step 04: Run all lexical methods" "log/04_lexical_methods.Rout"

# The Supervised Language Models require using a GPU. Many GPUs run out of memory over multiple model runs and CUDA makes it difficult
# to free up memory. So, the code below runs through models separately by training dataset while reseting the kernel. If a GPU is 
# sufficiently small, it migth be best to modify the shell script further to run these models by training set and model type.

# Define training sets
training_sets=('party' 'nominate' 'handcode')

#for i in "${!training_sets[@]}"; do
#    training_set="${training_sets[$i]}"
#    run_command "python -u code/05_pretrained_transformer_methods.py $training_set" "Step 05: Run Supervised Language models using training data: $training_set" "log/05_pretrained_transformer_methods_$training_set.log"
#done

# Only run GPT methods if desired by the user.
if [ "$RUN_GPT" = true ]; then
    run_command "bash code/06_submit_tuned_gpt.sh" "Step 06: Run GPT models"
else
    echo "Step 06: Skipping code for GPT models and using estimates from Harvard Dataverse. Use --run-gpt to replicate these models but make sure to provide API keys first!"
    echo ""
fi

run_command "Rscript --no-save --no-restore --verbose code/07_combine_results.R" "Step 07: Combine all estimates into result table" "log/07_combine_results.Rout"

run_command "Rscript --no-save --no-restore --verbose code/08_evaluate_results.R" "Step 08: Analyze and visualize results" "log/08_evaluate_results.Rout"

echo "========================================"
echo "✓ Analysis pipeline completed successfully!"