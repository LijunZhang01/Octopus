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
gpus=3
max_token=64
log_path="/home/zlj/Octopus/loglr5"
cd_alpha=2.5
amber_path="/home/zlj/Octopus/"
use_cd=False
use_m3id=False
json_path="/home/zlj/Octopus/data/train_all_idt"
data_path="/home/zlj/Octopus/dataset/images/val2014"

model_path="/home/zlj/Octopus/llava-v1.5-7b"
checkpoint_path="/home/zlj/Octopus/checkpoint/gendata_new_10000"
####################################################

result_json_path="${checkpoint_path}/Amber_result.json"

export CUDA_VISIBLE_DEVICES=${gpus}
python ./eval_bench/amber_gendata.py \
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
    --use_m3id ${use_m3id} \
    --inference_data \
    "/home/zlj/Octopus/log/Amber_result.json" \
    --word_association \
    "/home/zlj/Octopus//data/relation.json" \
    --safe_word \
    "/home/zlj/Octopus//data/safe_words.txt" \
    --annotation \
    "/home/zlj/Octopus//data/annotations.json" \
    --metrics \
    "/home/zlj/Octopus//data/metrics.txt" \
    --evaluation_type \
    a

#CUDA_VISIBLE_DEVICES=2 python ./eval_bench/amber_eval_${model}.py \
#    --seed 42 \
#    --model-path "/home/zlj/Octopus/llava-v1.5-7b" \
#    --json_path "/home/zlj/Octopus/data/query/query_all.json" \
#    --data_path "/home/zlj/Octopus/data/image" \
#    --log_path "/home/zlj/Octopus/log/" \
#    --conv-mode "llava_v1" \
#    --batch-size 1 \
#    --use_avisc True \
#    --layer_gamma 0.5 \
#    --masking_scheme "zeros" \
#    --lamb 1.0 \
#    --use_cd False\
#    --max_token 64 \
#    --cd_alpha 2.5 \
#    --use_m3id False \

# python ./experiments/AMBER/inference.py \
#     --inference_data ${result_json_path} \
#     --word_association "${amber_path}/data/relation.json" \
#     --safe_word "${amber_path}/data/safe_words.txt"\
#     --annotation "${amber_path}/data/annotations.json"\
#     --metrics "${amber_path}/data/metrics.txt"\
#     --evaluation_type a \

# python ./experiments/AMBER/inference.py \
#    --inference_data "/home/zlj/Octopus/log/Amber_result.json" \
#    --word_association "/home/zlj/Octopus//data/relation.json" \
#    --safe_word "/home/zlj/Octopus//data/safe_words.txt"\
#    --annotation "/home/zlj/Octopus//data/annotations.json"\
#    --metrics "/home/zlj/Octopus//data/metrics.txt"\
#    --evaluation_type a \

