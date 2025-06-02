#!/bin/bash

# Run pretrained models
models = ("gpt35", "gpt4", "gpt4o")

for m in "${models[@]}"; do
    echo "Submitting job: $m"
    Rscript gpt_methods.R "$m" "$s" > "logs/output_gpt_pretrained_${m}_recovery.log" 2>&1 &
done

# Run tuned models

models=("gpt35_tune_nominate" "gpt4o_tune_nominate")
subjects=("biden" "trump")

for m in "${models[@]}"; do
    for s in "${subjects[@]}"; do
        echo "Submitting job: $m $s"
        Rscript gpt_methods.R "$m" "$s" > "logs/output_gpt_${m}_${s}_recovery.log" 2>&1 &
    done
done
