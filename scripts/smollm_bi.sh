#!/bin/bash

# Set common variables
model=HuggingFaceTB/SmolLM-135M
bi_score='ranks/wikitext2/SmolLM-135M/'
output_dir='out/SmolLM-135M/structured/'
cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

run_python_command () {
    python main.py \
    --model $model \
    --prune_method $1 \
    --bi_score $bi_score \
    --sparsity_ratio $2 \
    --save $3 
}

for sparsity_ratio in 1
do
    echo "Running with BI score computation"
    run_python_command "bi" $sparsity_ratio $output_dir
    echo "Finished BI score computation"
done