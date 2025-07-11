#!/bin/bash

# Set common variables
model=meta-llama/Llama-2-7b-hf
bi_score='ranks/wikitext2/llama-2-7b/'
output_dir='out/llama-2-7b/'

# model=HuggingFaceTB/SmolLM-135M
# bi_score='ranks/wikitext2/SmolLM-135M/'
# output_dir='out/SmolLM-135M/'

# model=Qwen/Qwen2.5-0.5B
# bi_score='ranks/wikitext2/Qwen2.5-0.5B/'
# output_dir='out/Qwen2.5-0.5B/'

# model=Qwen/Qwen3-8B
# bi_score='ranks/wikitext2/Qwen3-8B/'
# output_dir='out/Qwen3-8B/'


tol=0.96
cuda_device=1
dataset=wikitext2

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $2 \
    --dataset $dataset \
    --bi_score $bi_score \
    --save $3 \
    --tol $4 \
    --nsamples 256 \
    --save_model $5 \
    --seed 42
}

for sparsity_ratio in 88 # 76 64 52 40
do
    save_path="${output_dir}${sparsity_ratio}"
    mkdir -p "$save_path"  
    echo "Running with flatllm pruning method"
    run_python_command "flatllm" $sparsity_ratio $output_dir $tol $save_path
    echo "Finished flatllm pruning method"
done