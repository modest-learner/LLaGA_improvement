#!/bin/bash

model_path="${MODEL_PATH:-./models/llaga-vicuna-7b-simteg-HO-general_model-2-layer-mlp-projector}"
model_base="${MODEL_BASE:-./models/vicuna-7b-v1.5-16k}" #meta-llama/Llama-2-7b-hf
mode="v1" # use 'llaga_llama_2' for llama and "v1" for others
dataset="${DATASET:-cora}" #test dataset
task="${TASK:-lp}" #test task
emb="${EMB:-simteg}"
use_hop="${USE_HOP:-4}"
sample_size="${SAMPLE_SIZE:-10}"
template="${TEMPLATE:-HO}" # or ND
output_path="${OUTPUT_PATH:-./results/official_ho_general/cora_lp.jsonl}"
input_mode="${INPUT_MODE:-graph}"
prompt_style="${PROMPT_STYLE:-official}"

python eval/eval_pretrain.py \
--model_path "${model_path}" \
--model_base "${model_base}" \
--conv_mode  ${mode} \
--dataset ${dataset} \
--pretrained_embedding_type ${emb} \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--answers_file "${output_path}" \
--task ${task} \
--cache_dir ../../checkpoint \
--template ${template} \
--input_mode "${input_mode}" \
--prompt_style "${prompt_style}" \
"$@"
