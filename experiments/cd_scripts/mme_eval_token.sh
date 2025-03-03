seed=55
dataset_name="mme"
# question_file="/path/to/the/neurips2024/experiments/data/MME_Benchmark_release_version/mme_hallucination.jsonl"
question_file="/home/zlj/AvisC-master/data/MME/MME_Benchmark_release_version/llava_mme.jsonl"
image_folder="/home/zlj/AvisC-master/data/MME/MME_Benchmark_release_version"

## max token maybe important
### set below ###
model="llava" # llava | qwenvl | instructblip
use_avisc=True
layer_gamma=0.5
masking_scheme="zeros"
lambda=1.0
lamb=1.0
max_token=64
use_cd=False
cd_alpha=1.0
gpus=0
use_m3id=False
model_path="/home/zlj/AvisC-master/llava-v1.5-7b"
log_path="/home/zlj/AvisC-master/log_mme_gen"
answer_path="${log_path}"
# checkpoint_path="/home/zlj/AvisC-master/checkpoint/0817_pope_coco/"
checkpoint_path="/home/zlj/AvisC-master/checkpoint/0808dpo_coco/"

#################




export CUDA_VISIBLE_DEVICES=${gpus}
python /home/zlj/AvisC-master/experiments/eval/eval_token_mme_llava.py \
--seed ${seed} \
--model-path ${model_path} \
--question-file ${question_file} \
--image-folder ${image_folder} \
--checkpoint_path ${checkpoint_path} \
--answers-file ${answer_path}/answer.jsonl \
--use_avisc ${use_avisc} \
--layer_gamma ${layer_gamma} \
--masking_scheme ${masking_scheme} \
--lamb ${lambda} \
--cd_alpha ${cd_alpha} \
--max_token ${max_token} \
--use_cd ${use_cd} \
--use_m3id ${use_m3id} \
--log_path ${log_path} \

python ./experiments/eval/convert_answer_to_mme.py \
--output_path ${answer_path}/answer.jsonl \
--seed ${seed} \
--log_path ${answer_path} \





python ./experiments/eval/calculation.py \
--results_dir ${answer_path}/mme_answers \


