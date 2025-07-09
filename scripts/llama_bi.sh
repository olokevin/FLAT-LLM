#!/bin/bash

# Set common variables
# model=meta-llama/Llama-2-7b-hf
# bi_score='ranks/wikitext2/llama-2-7b/'
# output_dir='out/llama_7b/structured/'

# model=HuggingFaceTB/SmolLM-135M
# bi_score='ranks/wikitext2/SmolLM-135M/'
# output_dir='out/SmolLM-135M/structured/'

model=Qwen/Qwen2.5-0.5B
bi_score='ranks/wikitext2/Qwen2.5-0.5B/'
output_dir='out/Qwen2.5-0.5B/structured/'

# model=Qwen/Qwen3-8B
# bi_score='ranks/wikitext2/Qwen3-8B/'
# output_dir='out/Qwen3-8B/structured/'

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