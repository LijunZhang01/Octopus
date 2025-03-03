import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from transformers.modeling_outputs import CausalLMOutputWithPast
from PIL import Image
import math
import torch.distributed as dist
from utils import dist_util
from utils.logger import create_logger
from glob import glob

# import kornia
from transformers import set_seed
from avisc_utils.vcd_add_noise import add_diffusion_noise
from avisc_utils.avisc_sample import evolve_avisc_sampling
evolve_avisc_sampling()

def recorder(out):
    NEG_WORDS = ["No", "not", "no", "NO"]

    out = out.replace('.', '')
    out = out.replace(',', '')
    words = out.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class MyModel(torch.nn.Module):
    def __init__(self, model, d_model=4096, nhead=2, num_decoder_layers=2, num_classes=4, n_query=4,bt=1):
        super(MyModel, self).__init__()
        self.n_query = n_query
        self.model = model
        self.Llama = model.model
        self.queries = torch.nn.Parameter(torch.randn(n_query, 4096).to(dtype=torch.float16))
        # self.mlp = torch.nn.Linear(4096, 4).to(dtype=torch.float16)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 4096).to(dtype=torch.float16))
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model, nhead).to(dtype=torch.float16)
        decoder_layer.apply(self.init_weights)
        self.transformer = torch.nn.TransformerDecoder(decoder_layer, num_decoder_layers).to(dtype=torch.float16)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, 1024).to(dtype=torch.float16),
            # torch.nn.ReLU(),
            torch.nn.LeakyReLU(),
            # torch.nn.Sigmoid(),
            torch.nn.Linear(1024, num_classes).to(dtype=torch.float16)
        )
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                torch.nn.init.constant_(layer.bias, 0)  # 偏置通常初始化为0
        self.bt=bt
    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            # 使用 He 初始化
            torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                # 初始化偏置为0
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, torch.nn.LayerNorm):
            # 层归一化层的初始化
            torch.nn.init.constant_(m.weight, 1.0)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self,
        input_ids= None,
        attention_mask= None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        images = None,
        images_cd = None,
        cd_beta = None,
        cd_alpha = None,
        img_idx = None,
        mask_idx = None,
        return_dict = None,
        kernel_size=None,
        use_avisc=None,
        layer_gamma=None,
        masking_scheme=None,
        lamb=None,
        question_id=None,
        use_m3id=None,
        is_eval=None,
        temp=None,):

        attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.long, device=input_ids.device)
        if input_ids is not None:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.model.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images)
        if mask_idx is not None and past_key_values is None:

            for input_embed, idx in zip(inputs_embeds, mask_idx):
                # input_embed[idx] = torch.randn(input_embed[idx].size(), dtype=input_embed.dtype).to(input_embed.device) * 0.1
                #input_embed[idx] = add_diffusion_noise(input_embed[idx], noise_step=500)
                if masking_scheme.lower() == "ones":
                    input_embed[idx + 35] = 1.0
                    # print("ones")
                elif masking_scheme.lower() == "zeros":
                    input_embed[idx + 35] = 0.0
                    # print("zeros")
                elif masking_scheme.lower() == "noise":
                    input_embed[idx + 35] = torch.randn(input_embed[idx + 35].size(), dtype=input_embed.dtype).to(input_embed.device)
                    # print("noise")
                else:
                    input_embed[idx + 35] = 0.0


        with torch.no_grad():
            outputs = self.Llama(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True
            )
        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
        else:
            output = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        inputs_transformer = torch.cat(
            (self.cls_token.unsqueeze(0).expand(self.bt, -1, -1), self.queries.unsqueeze(0).expand(self.bt, -1, -1)), dim=1)
        # inputs_transformer = inputs_transformer.to(dtype=torch.float32)
        out_transformer = self.transformer(inputs_transformer.transpose(0, 1),hidden_states.transpose(0, 1)).transpose(0, 1)

        logits_mlp = self.mlp(out_transformer)
        return logits_mlp[:, 0, :],output


    def generate(self,
        inputs=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        **kwargs):
        tt=self.model.generate(inputs,
        generation_config,
        logits_processor,
        stopping_criteria,
        prefix_allowed_tokens_fn,
        synced_gpus,
        assistant_model,
        streamer,
        mymodel=self,
        **kwargs)
        return tt
def eval_model(args):
    
    # set up gpu and logging
    
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        # if not os.path.exists(args.checkpoint_path):
        #     os.makedirs(args.checkpoint_path, exist_ok=True)
        #     os.makedirs(os.path.join(args.checkpoint_path, "log"), exist_ok=True)
        # args.log_path = os.path.join(args.checkpoint_path, "log")
        os.makedirs(
            args.log_path, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model_path.split("/")[-1]
        experiment_index = len(glob(f"{args.log_path}/{model_string_name}/*"))
        experiment_dir = f"{args.log_path}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
    logger.info(f"use_cd: {args.use_cd}, method: {args.use_avisc}, layer_gamma: {args.layer_gamma}, masking_scheme: {args.masking_scheme}, lamb: {args.lamb}")
    logger.info(f"question_file : {args.question_file}")


    
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    def load_model(model_path):
        model_test = MyModel(model).to(device)
        checkpoint = torch.load(model_path)
        model_test.queries.data = checkpoint['query']
        model_test.cls_token.data = checkpoint['cls']
        model_test.mlp.load_state_dict(checkpoint['mlp_state_dict'])
        model_test.transformer.load_state_dict(checkpoint['transformer'])
        # for param in model_test.gpt2.parameters():
        #     param.requires_grad = False
        return model_test

    # 加载保存的模型权重
    loaded_model = load_model(os.path.join(args.checkpoint_path, 'result.pth'))
    # 然后你可以使用 loaded_model 进行验证
    for line in tqdm(questions):
    # for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        
        
        # one word processing
        qs = qs.split('\n')[0] 
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        # print(image_tensor.shape)
        image_tensor = image_tensor.unsqueeze(0)

        # if args.use_cd:
        image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        # else:
        #     image_tensor_cd = None      

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            with torch.no_grad():
                input_token_len = input_ids.shape[1]
                output_ids = loaded_model.generate(input_ids,
                                             images=image_tensor.half().cuda(),
                                             images_cd=image_tensor_cd.half().cuda(),
                                             cd_alpha=args.cd_alpha,
                                             cd_beta=args.cd_beta,
                                             do_sample=True,
                                             temperature=args.temperature,
                                             top_p=args.top_p,
                                             top_k=args.top_k,
                                             max_new_tokens=args.max_token,
                                             use_cache=True,
                                             use_avisc=True,
                                             layer_gamma=args.layer_gamma,
                                             masking_scheme=args.masking_scheme,
                                             lamb=args.lamb,
                                             use_m3id=True,
                                             output_scores=True,
                                             output_attentions=True,
                                             is_eval=True,
                                             return_dict_in_generate=False, )
        # with torch.inference_mode():
        #     output_ids = model.generate(
        #         input_ids,
        #         images=image_tensor.unsqueeze(0).half().cuda(),
        #         images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
        #         cd_alpha = args.cd_alpha,
        #         cd_beta = args.cd_beta,
        #         do_sample=True,
        #         temperature=args.temperature,
        #         top_p=args.top_p,
        #         top_k=args.top_k,
        #         max_new_tokens=args.max_token,
        #         use_cache=True,
        #         use_avisc=args.use_avisc,
        #         layer_gamma=args.layer_gamma,
        #         masking_scheme=args.masking_scheme,
        #         lamb=args.lamb,
        #         use_m3id=args.use_m3id,
        #     )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        
        
        logger.info(f"[{image_file}]")
        logger.info(f"prompt: {cur_prompt}") 
        logger.info(f"text: {outputs}")  
        
        ## one word processing 
        outputs = recorder(outputs)          

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="path/checkpoints/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=" /path/to/experiments/data/MME_Benchmark_release_version")
    parser.add_argument("--question-file", type=str, default=" /path/to/experiments/data/MME_Benchmark_release_version/llava_mme.jsonl")
    parser.add_argument("--answers-file", type=str, default="path/logs/mme//test/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--log_path", type=str, default="path/logs/mme")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", type=str2bool, default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_avisc", type=str2bool, default=False)
    parser.add_argument("--layer_gamma", type=float, default=0.5)
    parser.add_argument("--masking_scheme", type=str, default="zeros")
    parser.add_argument("--lamb", type=float, default=100)
    parser.add_argument("--max_token", type=int, default=64)
    parser.add_argument("--use_m3id", type=str2bool, default=True)
    parser.add_argument("--checkpoint_path", type=str, default="/home/zlj/AvisC-master/checkpoint/try")
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
