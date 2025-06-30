#!/bin/bash

# Set common variables
model=Qwen/Qwen2.5-0.5B
bi_score='ranks/wikitext2/Qwen2.5-0.5B/'
cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

run_python_command () {
    python main_mistral.py \
    --model $model \
    --prune_method $1 \
    --bi_score $bi_score \
    --sparsity_ratio $2 \
    --save $3 
}

for sparsity_ratio in 1
do
    echo "Running with BI score computation"
    run_python_command "bi" $sparsity_ratio "out/Qwen2.5-0.5B/structured/"
    echo "Finished BI score computation"
done