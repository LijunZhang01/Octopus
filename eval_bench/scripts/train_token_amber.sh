#!/bin/bash


## set below
####################################################
seed=42
conv="llava_v1"
batch_size=1
model="llava" # llava | qwen-vl | instructblip
use_avisc=True
layer_gamma=0.5
masking_scheme="zeros"			# "10:1" or "-1"
lamb=1.0
gpus=0
max_token=64
log_path="/home/zlj/Octopus/log"
cd_alpha=2.5
amber_path="/home/zlj/Octopus/"
use_cd=False
use_m3id=False
json_path="/home/zlj/Octopus/data/train.json"
data_path="/home/zlj/Octopus/dataset/images/val2014"

model_path="/home/zlj/Octopus/llava-v1.5-7b"
####################################################


# checkpoint_path="/home/zlj/Octopus/checkpoint/0715dpo"
checkpoint_path="/home/zlj/Octopus/checkpoint/0123_shijian_sample"
result_json_path="${checkpoint_path}/Amber_result.json"
export CUDA_VISIBLE_DEVICES=${gpus}
python /home/zlj/Octopus/eval_bench/train_token_amber.py \
    --seed ${seed} \
    --model-path ${model_path} \
    --json_path ${json_path} \
    --data_path ${data_path} \
    --log_path ${log_path} \
    --checkpoint_path ${checkpoint_path} \
    --conv-mode ${conv} \
    --batch-size ${batch_size} \
    --use_avisc ${use_avisc} \
    --layer_gamma ${layer_gamma} \
    --masking_scheme ${masking_scheme} \
    --lamb ${lamb} \
    --use_cd ${use_cd} \
    --max_token ${max_token} \
    --cd_alpha ${cd_alpha} \
    --use_m3id ${use_m3id}
 # --result_json_path ${result_json_path}
