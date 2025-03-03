import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import LlamaModel, LlamaConfig
from utils import dist_util
from utils.logger import create_logger
from glob import glob
from llava.ppo_load import PPOTrainer
from PIL import Image
import math

from amber_loader_train import AMBERDataSet
from torch.utils.tensorboard import SummaryWriter
# import kornia
from transformers import set_seed
from avisc_utils.vcd_add_noise import add_diffusion_noise
from avisc_utils.avisc_sample import evolve_avisc_sampling
# from avisc_utils.avisc_sample_ppo import evolve_avisc_sampling
# from avisc_utils.mask import my_mask
# from avisc_utils.llama import llama,decoder_norm
evolve_avisc_sampling()
# my_mask()
# llama()
# decoder_norm()
torch.multiprocessing.set_sharing_strategy('file_system')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="AMBER-Adv evaluation on LVLMs.")
    parser.add_argument("--model-path", type=str, default="path/checkpoints/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)

    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--json_path", type=str, default="path/to/to/experiments/AMBER/data/query/query_all.json")
    parser.add_argument("--data_path", type=str, default="path/dataset/AMBER/image")
    parser.add_argument("--log_path", type=str, default="path/logs/amber")
    parser.add_argument("--checkpoint_path", type=str, default="/home/zlj/AvisC-master/checkpoint/try")

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", type=str2bool, default=False)
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=7, help="specify the gpu to load the model.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--use_avisc", type=str2bool, default=True)
    parser.add_argument("--layer_gamma", type=float, default=0.5)
    parser.add_argument("--masking_scheme", type=str, default="zeros")
    parser.add_argument("--lamb", type=float, default=0.99)
    parser.add_argument("--exp_description", type=str, default="..")
    parser.add_argument("--max_token", type=int, default=64)
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    parser.add_argument("--total_episodes", type=int, default=8)
    parser.add_argument("--n_ppo_epoch", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--word_association", type=str, default='')
    parser.add_argument("--safe_words", type=str, default='')
    parser.add_argument("--inference_data", type=str)
    parser.add_argument("--annotation", type=str, default='')
    parser.add_argument("--metrics", type=str, default='')
    parser.add_argument("--similarity_score", type=float, default=0.8)
    parser.add_argument('--evaluation_type', choices=['a', 'g', 'd', 'de', 'da', 'dr'],
                        help='a: all tasks and dimensions    g: generative task    d: descriminative task    de, da, dr: existence, attribute, relation')

    args = parser.parse_args()
    return args


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def recorder(out):
    NEG_WORDS = ["No", "not", "no", "NO"]

    out = out.replace('.', '')
    out = out.replace(',', '')
    words = out.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"

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


def main():
    args = parse_args()
    setup_seeds(args.seed)
    # Setup DDP:
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path, exist_ok=True)
            os.makedirs(os.path.join(args.checkpoint_path, "log"), exist_ok=True)
        args.log_path = os.path.join(args.checkpoint_path, "log")
        # os.makedirs(
        #     args.log_path, exist_ok=True
        # )  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model_path.split("/")[-1]
        experiment_dir = f"{args.log_path}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"exp_description: {args.exp_description}")
    else:
        logger = create_logger(None)

    # ========================================
    #             Model Initialization
    # ========================================
    print('Initializing Model')
    logger.info(
        f"use_cd: {args.use_cd}, method: {args.use_avisc}, layer_gamma: {args.layer_gamma}, masking_scheme: {args.masking_scheme}, lamb: {args.lamb}")

    #### for avisc
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    tokenizer.padding_side = "left"
    # load AMBER data
    # json_path : json path, e.g. data/AMBER/coco/coco_AMBER_random.json
    # data_path : image folder path e.g. data/coco/images/val2024

    AMBER_dataset = AMBERDataSet(
        json_path=args.json_path,
        data_path=args.data_path,
        trans=image_processor,
        model='llava',
        num_gen=10000,
        num_dis=0
    )
    AMBER_loader = torch.utils.data.DataLoader(
        AMBER_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    print("load data finished")

    print("Start eval...")
    result_json_path = os.path.join(args.checkpoint_path, "Amber_result.json")

    result = []

    # ans={}
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.size()}")
    #     ans[name]=param.size()
    # with open("model_arch.json", 'w', encoding='utf-8') as f:
    #     json.dump(ans, f, ensure_ascii=False, indent=4)


    policy = MyModel(model).to(device)
    # Freeze the GPT model
    for param in policy.Llama.parameters():
        param.requires_grad = False
    for param in policy.model.parameters():
        param.requires_grad = False

    # Create an optimizer that only updates the parameters that require gradients
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=1e-5, eps=1e-4)
    # optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=1e-5)

    # scheduler = transformers.get_scheduler(
    #     name='linear',
    #     optimizer=optimizer1,
    #     num_warmup_steps=100 * args.n_ppo_epoch,
    #     num_training_steps=total_steps * args.n_ppo_epoch,
    # )
    # trainer = PPOTrainer(
    #     args=args,
    #     train_dataloader=AMBER_loader,
    #     ref_policy_model=ref_policy,
    #     policy_model=policy,
    #     tokenizer=tokenizer,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    # )
    # for i in range(5):
    #     for batch_id, data in tqdm(enumerate(AMBER_loader), total=len(AMBER_loader)):
    #         # trainer.train(data,batch_id*(i+1))
    #         trainer.train(data, len(AMBER_loader) * i + batch_id)
    #         for name, parms in policy.named_parameters():
    #             if "model." not in name and 'cls' not in name:
    #                 print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
    #                       ' -->grad_value:', torch.mean(parms.grad))
    #     torch.save({
    #         'query': policy.queries.data,
    #         'cls': policy.cls_token.data,
    #         'mlp_state_dict': policy.mlp.state_dict(),
    #         'transformer': policy.transformer.state_dict()
    #     }, os.path.join(args.checkpoint_path, 'result_{}.pth'.format(i)))
    # # steps = list(range(total_steps + 1))
    # # steps = tqdm(steps)
    # # for step in steps:
    # #     trainer.train(step)
    #
    # # 保存 query 和 MLP 的权重
    # torch.save({
    #     'query': policy.queries.data,
    #     'cls': policy.cls_token.data,
    #     'mlp_state_dict': policy.mlp.state_dict(),
    #     'transformer': policy.transformer.state_dict()
    # }, os.path.join(args.checkpoint_path, 'result.pth'))
    #
    # # 验证时加载模型权重
    # def load_model(model_path):
    #     model_test = MyModel(model).to(device)
    #     checkpoint = torch.load(model_path)
    #     model_test.queries.data = checkpoint['query']
    #     model_test.cls_token.data = checkpoint['cls']
    #     model_test.mlp.load_state_dict(checkpoint['mlp_state_dict'])
    #     model_test.transformer.load_state_dict(checkpoint['transformer'])
    #     # for param in model_test.gpt2.parameters():
    #     #     param.requires_grad = False
    #     return model_test
    #
    # # 加载保存的模型权重
    # loaded_model = load_model(os.path.join(args.checkpoint_path, 'result.pth'))
    # # 然后你可以使用 loaded_model 进行验证
    writer = SummaryWriter(args.checkpoint_path)
    for epoch in range(args.epochs):
        print("=="*20)
        print("第{}轮训练".format(epoch))
        print("==" * 20)
        for batch_id, data in tqdm(enumerate(AMBER_loader), total=len(AMBER_loader)):

            optimizer.zero_grad()
            image = data["image"]
            qs = data["query"]
            ids = data["id"]
            y_w=data['y_w']
            y_l=data['y_l']
            image_path = data["image_path"]

            # ==============================================
            #             Text prompt setting
            # ==============================================
            if model.config.mm_use_im_start_end:
                qu = [DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + _ for _ in qs]
            else:
                qu = [DEFAULT_IMAGE_TOKEN + '\n' + _ for _ in qs]

            input_ids = []

            for i in range(args.batch_size):
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qu[i])
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                # ==============================================
                #             Image tensor setting
                # ==============================================

                input_id = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                    0).cuda()

                input_ids.append(
                    input_id
                )

            def make_batch(input_ids):
                input_ids = [_.squeeze(0) for _ in input_ids]
                max_len = max([_.shape[0] for _ in input_ids])
                input_ids = [torch.cat([torch.zeros(max_len - _.shape[0], dtype=torch.long).cuda(), _], dim=0) for _ in
                             input_ids]
                return torch.stack(input_ids, dim=0)

            input_ids = make_batch(input_ids)
            image_tensor = image

            # ==============================================
            #             avisc method setting
            # ==============================================
            image_tensor_cd = add_diffusion_noise(image_tensor, noise_step=500)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # with torch.inference_mode():
            #     with torch.no_grad():
            #         output_ids = model.generate(input_ids,
            #                                     images=image_tensor.half().cuda(),
            #                                     images_cd=None,
            #                                     cd_alpha=args.cd_alpha,
            #                                     cd_beta=args.cd_beta,
            #                                     do_sample=True,
            #                                     temperature=args.temperature,
            #                                     top_p=args.top_p,
            #                                     top_k=args.top_k,
            #                                     max_new_tokens=args.max_token,
            #                                     use_cache=False,
            #                                     use_avisc=False,
            #                                     layer_gamma=args.layer_gamma,
            #                                     masking_scheme=args.masking_scheme,
            #                                     lamb=args.lamb,
            #                                     use_m3id=False,
            #                                     output_scores=True,
            #                                     output_attentions=True,
            #                                     is_eval=True,
            #                                     output_hidden_states=True)

            # with torch.enable_grad():
            output_ids = policy.generate(input_ids,
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
                                        is_eval=False,
                                        return_dict_in_generate=True,)
            text_scores=torch.stack(output_ids['scores'][0], dim=1)
            action_scores=torch.stack(output_ids['scores'][1], dim=1)
            policy_chosen_label = y_w.to(dtype=torch.int64,device=action_scores.device)
            policy_rejected_label = y_l.to(dtype=torch.int64,device=action_scores.device)
            # reference_chosen_label = torch.randint(0, 4, (action_scores.shape[0], action_scores.shape[1]),
            #                                     dtype=torch.int64).to(device=action_scores.device)
            # reference_rejected_label = torch.randint(0, 4, (action_scores.shape[0], action_scores.shape[1]),
            #                                     dtype=torch.int64).to(device=action_scores.device)
            len_action=action_scores.shape[1]
            len_text=min(policy_chosen_label.shape[1],policy_rejected_label.shape[1])
            if len_action<len_text:
                policy_chosen_label=policy_chosen_label[:,:len_action]
                policy_rejected_label = policy_rejected_label[:, :len_action]
            if len_action>=len_text:
                action_scores=action_scores[:,:len_text,:]
                policy_chosen_label = policy_chosen_label[:, :len_text]
                policy_rejected_label = policy_rejected_label[:, :len_text]

            policy_chosen_logps=torch.gather(action_scores.log_softmax(-1), 2, policy_chosen_label[:, :, None]).squeeze(2).sum(-1)
            policy_rejected_logps=torch.gather(action_scores.log_softmax(-1), 2, policy_rejected_label[:, :, None]).squeeze(2).sum(-1)
            # reference_chosen_logps = torch.gather(action_scores.log_softmax(-1), 2, reference_chosen_label[:, :, None]).squeeze(2).sum(-1)
            # reference_rejected_logps = torch.gather(action_scores.log_softmax(-1), 2, reference_rejected_label[:, :, None]).squeeze(2).sum(-1)

            pi_logratios = policy_chosen_logps - policy_rejected_logps
            # ref_logratios = reference_chosen_logps - reference_rejected_logps
            # if reference_free:
            ref_logratios = 0
            logits = (pi_logratios - ref_logratios).sum(-1)
            losses = -F.logsigmoid(args.beta * logits)
            writer.add_scalar('training loss', losses.item(), len(AMBER_loader)*epoch+batch_id)
            losses.requires_grad_(True)
            losses.backward()
            optimizer.step()
        torch.save({
            'query': policy.queries.data,
            'cls': policy.cls_token.data,
            'mlp_state_dict': policy.mlp.state_dict(),
            'transformer': policy.transformer.state_dict()
        }, os.path.join(args.checkpoint_path, 'result_{}.pth'.format(epoch)))
    torch.save({
        'query': policy.queries.data,
        'cls': policy.cls_token.data,
        'mlp_state_dict': policy.mlp.state_dict(),
        'transformer': policy.transformer.state_dict()
    }, os.path.join(args.checkpoint_path, 'result.pth'))
    #     input_token_len = input_ids.shape[1]
    #     n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    #     if n_diff_input_output > 0:
    #         print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    #     outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    #     outputs = [_.strip() for _ in outputs]
    #     outputs = [_[:-len(stop_str)] if _.endswith(stop_str) else _ for _ in outputs]
    #
    #     for ip, q, a in zip(image_path, qs, outputs):
    #         logger.info(f"[{ip}]")
    #         logger.info(f"Q: {q}")
    #         logger.info(f"A: {a}")
    #
    #     for batch_id in range(len(ids)):
    #         if ids[batch_id] > 1004:
    #             outputs[batch_id] = recorder(outputs[batch_id])
    #
    #     for id, a in zip(ids, outputs):
    #         item = {
    #             "id": int(id),
    #             "response": a
    #         }
    #         result.append(item)
    #
    # with open(result_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)

    # 验证时加载模型权重



    ###########################################################
    #                        验证模型
    ###########################################################
    # def load_model(model_path):
    #     model_test = MyModel(model).to(device)
    #     checkpoint = torch.load(model_path)
    #     model_test.queries.data = checkpoint['query']
    #     model_test.cls_token.data = checkpoint['cls']
    #     model_test.mlp.load_state_dict(checkpoint['mlp_state_dict'])
    #     model_test.transformer.load_state_dict(checkpoint['transformer'])
    #     # for param in model_test.gpt2.parameters():
    #     #     param.requires_grad = False
    #     return model_test

    # # 加载保存的模型权重
    # loaded_model = load_model(os.path.join(args.checkpoint_path, 'result.pth'))
    # # 然后你可以使用 loaded_model 进行验证

    # for batch_id, data in tqdm(enumerate(AMBER_loader), total=len(AMBER_loader)):
    #     image = data["image"]
    #     qs = data["query"]
    #     ids = data["id"]
    #     image_path = data["image_path"]

    #     # ==============================================
    #     #             Text prompt setting
    #     # ==============================================
    #     if model.config.mm_use_im_start_end:
    #         qu = [DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + _ for _ in qs]
    #     else:
    #         qu = [DEFAULT_IMAGE_TOKEN + '\n' + _ for _ in qs]

    #     input_ids = []

    #     for i in range(args.batch_size):
    #         conv = conv_templates[args.conv_mode].copy()
    #         conv.append_message(conv.roles[0], qu[i])
    #         conv.append_message(conv.roles[1], None)
    #         prompt = conv.get_prompt()

    #         # ==============================================
    #         #             Image tensor setting
    #         # ==============================================

    #         input_id = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
    #             0).cuda()

    #         input_ids.append(
    #             input_id
    #         )

    #     def make_batch(input_ids):
    #         input_ids = [_.squeeze(0) for _ in input_ids]
    #         max_len = max([_.shape[0] for _ in input_ids])
    #         input_ids = [torch.cat([torch.zeros(max_len - _.shape[0], dtype=torch.long).cuda(), _], dim=0) for _ in
    #                      input_ids]
    #         return torch.stack(input_ids, dim=0)

    #     input_ids = make_batch(input_ids)
    #     image_tensor = image

    #     # ==============================================
    #     #             avisc method setting
    #     # ==============================================
    #     image_tensor_cd = add_diffusion_noise(image_tensor, noise_step=500)

    #     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    #     keywords = [stop_str]
    #     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


    #     with torch.inference_mode():
    #         with torch.no_grad():
    #             input_token_len = input_ids.shape[1]
    #             output_ids = loaded_model.generate(input_ids,
    #                                          images=image_tensor.half().cuda(),
    #                                          images_cd=image_tensor_cd.half().cuda(),
    #                                          cd_alpha=args.cd_alpha,
    #                                          cd_beta=args.cd_beta,
    #                                          do_sample=True,
    #                                          temperature=args.temperature,
    #                                          top_p=args.top_p,
    #                                          top_k=args.top_k,
    #                                          max_new_tokens=args.max_token,
    #                                          use_cache=True,
    #                                          use_avisc=True,
    #                                          layer_gamma=args.layer_gamma,
    #                                          masking_scheme=args.masking_scheme,
    #                                          lamb=args.lamb,
    #                                          use_m3id=True,
    #                                          output_scores=True,
    #                                          output_attentions=True,
    #                                          is_eval=True,
    #                                          return_dict_in_generate=False, )
    #     input_token_len = input_ids.shape[1]
    #     n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    #     if n_diff_input_output > 0:
    #         print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    #     outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    #     outputs = [_.strip() for _ in outputs]
    #     outputs = [_[:-len(stop_str)] if _.endswith(stop_str) else _ for _ in outputs]

    #     for ip, q, a in zip(image_path, qs, outputs):
    #         logger.info(f"[{ip}]")
    #         logger.info(f"Q: {q}")
    #         logger.info(f"A: {a}")

    #     for batch_id in range(len(ids)):
    #         if ids[batch_id] > 1004:
    #             outputs[batch_id] = recorder(outputs[batch_id])

    #     for id, a in zip(ids, outputs):
    #         item = {
    #             "id": int(id),
    #             "response": a
    #         }
    #         result.append(item)

    # with open(result_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
