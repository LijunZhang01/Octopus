# *********************************************************
#  Version 1
#  Author: Yushi Hu
#  Date: 2023-06-20
#  Description: the PPO trainer
#  Referenced: https://github.com/liujch1998/rainier/
#  All Rights Reserved.
#  *********************************************************

import argparse
from collections import defaultdict
from itertools import chain
import json
import logging
import numpy as np
import os
import nltk
from nltk.stem import WordNetLemmatizer
import random
import shutil
from tqdm import tqdm
from typing import Dict
import yaml
import nltk
from typing import Optional, List, Iterable, Dict, Any, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from avisc_utils.vcd_add_noise import add_diffusion_noise
import transformers
import accelerate
from torch.cuda.amp import GradScaler, autocast
# import wandb
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# from .utils import ensure_dir, set_seed, reduce_mean, reduce_sum, ceil_div, whiten, clamp
import spacy
nlp = spacy.load("en_core_web_lg")



class PPOTrainer:
    def __init__(self,
                 args: argparse.Namespace,
                 train_dataloader: DataLoader,
                 ref_policy_model,
                 policy_model,
                 tokenizer,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler
                 ):


        self.args = args
        self.train_dataloader = train_dataloader
        self.ref_policy_model = ref_policy_model
        self.policy_model = policy_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        # early stopping if KL too big
        self.should_early_stop = False
        self.huge_kl_count = 0

        self.batchify = lambda x, n: [x[i:i + n] for i in range(0, len(x), n)]

        # if self.accelerator.is_main_process:
        #     if args['logging']['wandb_log']:
        #         wandb.init(entity=args["logging"]["wandb_entity"], project=args["logging"]["wandb_project"],
        #                    name=args['logging']['run_name'], config=args)
        #     else:
        #         wandb.init(config=args, mode='disabled')
        #
        #     wandb.define_metric('train/step')
        #     wandb.define_metric('eval/step')
        #     wandb.define_metric('train/*', step_metric='train/step')
        #     wandb.define_metric('eval/*', step_metric='eval/step', summary='max')

        self.train_sampler = iter(self.train_dataloader)
        for _ in range(len(self.train_dataloader)):
            next(self.train_sampler)

        # self.eval_accs = {}

    def compute_advantages(self, results, num_samples):

        old_values = results['generated_value']
        rewards = results['rewards/penalized']
        mask = results['generated_attention_mask']  # (B, KL)

        with torch.no_grad():
            if self.args['ppo']['whiten_rewards']:
                whitened_rewards = whiten(rewards, mask, shift_mean=False, accelerator=self.accelerator)
            else:
                whitened_rewards = rewards

            lastgaelam = 0
            advantages_reversed = []
            gen_length = mask.sum(dim=1).max().item()
            for t in reversed(range(gen_length)):
                nextvalues = old_values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = whitened_rewards[:, t] + self.args['ppo']['gamma'] * nextvalues - old_values[:, t]
                lastgaelam = delta + self.args['ppo']['gamma'] * self.args['ppo']['lam'] * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)
            advantages = F.pad(advantages, (0, whitened_rewards.size(1) - gen_length), value=0.0)
            returns = advantages + old_values

            whitened_advantages = advantages.detach()
            whitened_advantages = whiten(advantages, mask, accelerator=self.accelerator).detach()

        results['whitened_advantages'] = whitened_advantages
        results['returns'] = returns

    def loss(self, results, all_mask_weight):

        old_values = results['generated_value']
        old_logprobs = results['generated_logprobs']
        mask = results['generated_attention_mask']  # (B, KL)

        whitened_advantages = results['whitened_advantages']
        returns = results['returns']

        weight = mask.sum(dim=1).float().mean().item() / all_mask_weight

        forward_inputs = {
            'prompts_input_ids': results['prompts_input_ids'],
            'prompts_attention_mask': results['prompts_attention_mask'],
            'generated_input_ids': results['generated_input_ids'],
            'generated_attention_mask': results['generated_attention_mask'],
        }

        policy_forward = self.policy_model.forward_pass(**forward_inputs)
        new_logprobs = policy_forward['generated_logprobs']

        ratio = torch.exp(new_logprobs - old_logprobs)
        pg_losses1 = -whitened_advantages * ratio
        pg_losses2 = -whitened_advantages * torch.clamp(ratio, min=1.0 - self.args['ppo']['cliprange'],
                                                        max=1.0 + self.args['ppo']['cliprange'])
        pg_loss = reduce_mean(torch.max(pg_losses1, pg_losses2), mask)
        pg_loss = pg_loss * weight

        if self.args['model']['value_model']['policy_value_sharing']:
            new_values = policy_forward['generated_value']
        else:
            value_forward = self.value_model.forward_pass(**forward_inputs)
            new_values = value_forward['generated_value']
            new_values *= mask

        new_values_clipped = clamp(new_values, old_values - self.args['ppo']['cliprange_value'],
                                   old_values + self.args['ppo']['cliprange_value'])
        vf_losses1 = torch.square(new_values - returns)
        vf_losses2 = torch.square(new_values_clipped - returns)
        vf_loss = .5 * reduce_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_loss = vf_loss * weight

        loss = self.args['ppo']['pg_coef'] * pg_loss + self.args['ppo']['vf_coef'] * vf_loss

        results['loss/total'] = loss
        results['loss/policy'] = pg_loss
        results['loss/value'] = vf_loss

    def train(self, step):

        #0.数据准备
        #------------------------------------------------------------
        # ------------------------------------------------------------
        try:
            data = next(self.train_sampler)
        except StopIteration:
            self.train_sampler = iter(self.train_dataloader)
            data = next(self.train_sampler)
        # data = next(self.train_dataloader)
        image = data["image"]
        qs = data["query"]
        ids = data["id"]
        image_path = data["image_path"]
        # ==============================================
        #             Text prompt setting
        # ==============================================
        if self.ref_policy_model.model.config.mm_use_im_start_end:
            qu = [DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + _ for _ in qs]
        else:
            qu = [DEFAULT_IMAGE_TOKEN + '\n' + _ for _ in qs]

        input_ids = []

        for i in range(self.args.batch_size):
            conv = conv_templates[self.args.conv_mode].copy()
            conv.append_message(conv.roles[0], qu[i])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        # ==============================================
        #             Image tensor setting
        # ==============================================
            input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            input_ids.append(
                input_id
            )
        def make_batch(input_ids):
            input_ids = [_.squeeze(0) for _ in input_ids]
            max_len = max([_.shape[0] for _ in input_ids])
            input_ids = [torch.cat([torch.zeros(max_len - _.shape[0], dtype=torch.long).cuda(), _], dim=0) for _ in input_ids]
            return torch.stack(input_ids, dim=0)
        input_ids = make_batch(input_ids)
        image_tensor = image
        # ==============================================
        #             avisc method setting
        # ==============================================
        image_tensor_cd = add_diffusion_noise(image_tensor, noise_step=500)
        # else:
        #     image_tensor_cd = None
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        # ------------------------------------------------------------
        # ------------------------------------------------------------

        # 1.按照policy_model得到需要选择那个action
        softmax=self.policy_model(input_ids, images=image_tensor.half().cuda())
        action = torch.argmax(softmax, dim=-1)
        # 2.得到action以后利用ref_policy_model得到这个action对应的预测结果 y_a
        print("action:",action[0])
        sequences_dict={}
        scores_dict={}
        mask_dict={}
        response_dict={}
        if True:
            with torch.inference_mode():
                with torch.no_grad():
                    input_token_len = input_ids.shape[1]
                    output_ids=self.ref_policy_model.generate(input_ids,
                            images=image_tensor.half().cuda(),
                            images_cd=None,
                            cd_alpha=self.args.cd_alpha,
                            cd_beta=self.args.cd_beta,
                            do_sample=True,
                            temperature=self.args.temperature,
                            top_p=self.args.top_p,
                            top_k=self.args.top_k,
                            max_new_tokens=self.args.max_token,
                            use_cache=True,
                            use_avisc=False,
                            layer_gamma=self.args.layer_gamma,
                            masking_scheme=self.args.masking_scheme,
                            lamb=self.args.lamb,
                            use_m3id=False,
                            output_scores=True,
                            output_attentions=True,
                            return_dict_in_generate=True,
                            output_hidden_states=True)
                    sequences = output_ids['sequences'][:, input_token_len:]
                    # scores = output_ids['scores']
                    scores = output_ids['scores']
                    scores = torch.nn.functional.softmax(torch.stack(scores, dim=1), dim=-1)
                    mask = sequences != 0
                    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                    # if n_diff_input_output > 0:
                    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                    response = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
                    response = [_.strip() for _ in response]
                    response = [_[:-len(stop_str)] if _.endswith(stop_str) else _ for _ in response]
                    sequences_dict[0]=sequences
                    scores_dict[0]=scores
                    mask_dict[0]=mask
                    response_dict[0]=response
        if True:
            with torch.inference_mode():
                with torch.no_grad():
                    input_token_len = input_ids.shape[1]
                    output_ids = self.ref_policy_model.generate(input_ids,
                            images=image_tensor.half().cuda(),
                            images_cd=None,
                            cd_alpha=self.args.cd_alpha,
                            cd_beta=self.args.cd_beta,
                            do_sample=True,
                            temperature=self.args.temperature,
                            top_p=self.args.top_p,
                            top_k=self.args.top_k,
                            max_new_tokens=self.args.max_token,
                            use_cache=True,
                            use_avisc=False,
                            layer_gamma=self.args.layer_gamma,
                            masking_scheme=self.args.masking_scheme,
                            lamb=self.args.lamb,
                            use_m3id=True,
                            output_scores=True,
                            output_attentions=True,
                            return_dict_in_generate=True,
                            output_hidden_states=True)
                    sequences = output_ids['sequences'][:, input_token_len:]
                    mask = sequences != 0
                    # scores = output_ids['scores']
                    scores = output_ids['scores']
                    scores = torch.nn.functional.softmax(torch.stack(scores, dim=1), dim=-1)
                    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                    # if n_diff_input_output > 0:
                    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                    response = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
                    response = [_.strip() for _ in response]
                    response = [_[:-len(stop_str)] if _.endswith(stop_str) else _ for _ in response]
                    sequences_dict[1] = sequences
                    scores_dict[1] = scores
                    mask_dict[1] = mask
                    response_dict[1] = response
        if True:
            with torch.inference_mode():
                with torch.no_grad():
                    input_token_len = input_ids.shape[1]
                    output_ids = self.ref_policy_model.generate(input_ids,
                            images=image_tensor.half().cuda(),
                            images_cd=None,
                            cd_alpha=self.args.cd_alpha,
                            cd_beta=self.args.cd_beta,
                            do_sample=True,
                            temperature=self.args.temperature,
                            top_p=self.args.top_p,
                            top_k=self.args.top_k,
                            max_new_tokens=self.args.max_token,
                            use_cache=True,
                            use_avisc=True,
                            layer_gamma=self.args.layer_gamma,
                            masking_scheme=self.args.masking_scheme,
                            lamb=self.args.lamb,
                            use_m3id=False,
                            output_scores=True,
                            output_attentions=True,
                            return_dict_in_generate=True,
                            output_hidden_states=True)
                    sequences = output_ids['sequences'][:, input_token_len:]
                    # scores = output_ids['scores']
                    scores = output_ids['scores']
                    scores = torch.nn.functional.softmax(torch.stack(scores, dim=1), dim=-1)
                    mask = sequences != 0
                    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                    # if n_diff_input_output > 0:
                    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                    response = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
                    response = [_.strip() for _ in response]
                    response = [_[:-len(stop_str)] if _.endswith(stop_str) else _ for _ in response]
                    sequences_dict[2] = sequences
                    scores_dict[2] = scores
                    mask_dict[2] = mask
                    response_dict[2] = response
        if True:
            with torch.inference_mode():
                with torch.no_grad():
                    input_token_len = input_ids.shape[1]
                    output_ids = self.ref_policy_model.generate(input_ids,
                            images=image_tensor.half().cuda(),
                            images_cd=image_tensor_cd.half().cuda(),
                            cd_alpha=self.args.cd_alpha,
                            cd_beta=self.args.cd_beta,
                            do_sample=True,
                            temperature=self.args.temperature,
                            top_p=self.args.top_p,
                            top_k=self.args.top_k,
                            max_new_tokens=self.args.max_token,
                            use_cache=True,
                            use_avisc=False,
                            layer_gamma=self.args.layer_gamma,
                            masking_scheme=self.args.masking_scheme,
                            lamb=self.args.lamb,
                            use_m3id=False,
                            output_scores=True,
                            output_attentions=True,
                            return_dict_in_generate=True,
                            output_hidden_states=True)
                    sequences = output_ids['sequences'][:, input_token_len:]
                    # scores = output_ids['scores']
                    scores = output_ids['scores']
                    scores = torch.nn.functional.softmax(torch.stack(scores, dim=1), dim=-1)
                    mask = sequences != 0

                    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                    # if n_diff_input_output > 0:
                    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                    response = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
                    response= [_.strip() for _ in response]
                    response= [_[:-len(stop_str)] if _.endswith(stop_str) else _ for _ in response]
                    sequences_dict[3] = sequences
                    scores_dict[3] = scores
                    mask_dict[3] = mask
                    response_dict[3] = response
        # 3.得到没用这个action的预测结果 y_o
        with torch.inference_mode():
            with torch.no_grad():
                input_token_len = input_ids.shape[1]
                output_ids = self.ref_policy_model.generate(input_ids,
                            images=image_tensor.half().cuda(),
                            images_cd=None,
                            cd_alpha=self.args.cd_alpha,
                            cd_beta=self.args.cd_beta,
                            do_sample=True,
                            temperature=self.args.temperature,
                            top_p=self.args.top_p,
                            top_k=self.args.top_k,
                            max_new_tokens=self.args.max_token,
                            use_cache=True,
                            use_avisc=False,
                            layer_gamma=self.args.layer_gamma,
                            masking_scheme=self.args.masking_scheme,
                            lamb=self.args.lamb,
                            use_m3id=False,
                            output_scores=True,
                            output_attentions=True,
                            return_dict_in_generate=True,
                            output_hidden_states=True)
                scores_old = output_ids['scores']
                scores_old = torch.nn.functional.softmax(torch.stack(scores_old, dim=1),dim=-1)

                sequences_old = output_ids['sequences'][:, input_token_len:]
                mask_old = sequences_old != 0
                response_old = self.tokenizer.batch_decode(sequences_old, skip_special_tokens=True)
                response_old = [_.strip() for _ in response_old]
                response_old = [_[:-len(stop_str)] if _.endswith(stop_str) else _ for _ in response_old]
                ref_logprobs = torch.gather(scores_old, 2, sequences_old[:, :, None]).squeeze(2)
        # 3.1 统一ref_logprobs和logprobs的大小和mask
        result_dict={}
        for key, value in sequences_dict.items():
            sequences=sequences_dict[key]
            scores=scores_dict[key]
            mask=mask_dict[key]
            response=response_dict[key]

            ol = sequences_old.size(1)
            ori = sequences.size(1)
            if ol < ori:
                padding = sequences.size(1) - sequences_old.size(1)
                sequences_old = F.pad(sequences_old, (0, padding), "constant", 0)
                mask_old = F.pad(mask_old, (0, padding), "constant", 0)
                ref_logprobs = F.pad(ref_logprobs, (0, padding), "constant", 0)
                # scores_old = F.pad(scores_old, (0, sequences_old.size(1)-scores_old.size(1)), "constant", 0)
            elif ol > ori:
                sequences_old = sequences_old[:, :sequences.size(1)]
                mask_old = mask_old[:, :sequences.size(1)]
                ref_logprobs = ref_logprobs[:, :sequences.size(1)]
                # scores_old = scores_old[:, :sequences.size(1)]



            # 4.计算 y_a 和 y_o 的概率差
            logprobs = torch.gather(scores, 2, sequences_old[:,:,None]).squeeze(2)
            mask_bin = mask & mask_old
            kl = ref_logprobs - logprobs
            kl[~mask_bin] = 0
            # 5.y_a得到这个句子的幻觉率，y_o得到原始的幻觉率，这两个差值相当于reward model的输出
            hal_old = []
            hal = []
            for i in range(len(ids)):
                hal_old.append(self.chair(ids[i],response_old[i]))
                hal.append(self.chair(ids[i],response[i]))
            # hal_old = torch.cat(hal_old, dim=0)
            # hal = torch.cat(hal, dim=0)
            # 6.按照以前ppo的说法，得到每个reward_i
            not_mask_bin = ~mask_bin.int()
            first_false_indices = torch.argmax(not_mask_bin, dim=1)
            for i in range(len(kl)):
                kl[i][first_false_indices[i]] += hal_old[i] -hal[i]
            print("chair_cha:",hal_old[0] -hal[0])
            # 7.累加和得到return_i,优势函数是否需要再相减，先不减吧
            # 反转每行
            kl_reversed = torch.flip(kl, dims=[1])
            # 计算反转后每行的累加和
            kl_cumsum_reversed = torch.cumsum(kl_reversed, dim=1)
            # 再次反转累加和结果以得到所需的后续累加和
            reward = torch.flip(kl_cumsum_reversed, dims=[1])
            # 8.计算损失，概率和return的累加和
            actor_loss = logprobs * reward
            result = torch.sum(actor_loss)
            result_dict[key]=result

        # result.requires_grad_(True)
        # 9.反向传播
        for ppo_epoch_idx in range(self.args.n_ppo_epoch):
            self.optimizer.zero_grad()
            # loss = torch.zeros(1,device=result.device)
            # loss.requires_grad_(True)
            # # loss=loss+result
            #
            # for i in range(softmax.size(1)):  # softmax.size(1) 是列数，这里是 4
            #     # 将索引转换为字符串，以匹配字典的键
            #     loss = loss + softmax[0, i] * result_dict[i]
            tt=[]
            for key,value in result_dict.items():
                tt.append(value.detach().item())
            expected_rewards = torch.sum(softmax * torch.tensor(tt).to(device=softmax.device), dim=1)
            # 最大化期望奖励，即最小化负的期望奖励
            loss = -torch.mean(expected_rewards*0.1)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.scheduler.step()

        # if step % self.args['train']['eval_interval'] == 0:
        #     self.save(step=step)
        #     self.valid(step=step)

        # self.accelerator.wait_for_everyone()
        # batch = next(self.train_sampler)
        #
        # self.ref_policy_model.model.eval()
        #
        # self.policy_model.model.eval()
        # # self.policy_model.model.train()
        #
        # self.value_model.model.eval()
        # self.value_model.linear.eval()
        # # self.value_model.model.train()
        #
        # # Rollout from current policy
        # with torch.no_grad():
        #     results = self.policy_model.sample(
        #         prompts_input_ids=batch['prompts_input_ids'],
        #         prompts_attention_mask=batch['prompts_attention_mask'],
        #         num_return_sequences=self.args['env']['train_num_samples_per_input'],
        #         **self.args['model']['policy_model']['train_generation_kwargs'],
        #     )
        #
        # forward_inputs = {
        #     'prompts_input_ids': results['prompts_input_ids'],
        #     'prompts_attention_mask': results['prompts_attention_mask'],
        #     'generated_input_ids': results['generated_input_ids'],
        #     'generated_attention_mask': results['generated_attention_mask'],
        # }
        #
        # with torch.no_grad():
        #     policy_forward = self.policy_model.forward_pass(**forward_inputs)
        #     results.update(policy_forward)
        #
        # # Run value network
        # if not self.args['model']['value_model']['policy_value_sharing']:
        #     with torch.no_grad():  # treat the values at beginning of step as ground-truth
        #         value_forward = self.value_model.forward_pass(**forward_inputs)
        #         results['generated_value'] = value_forward['generated_value']
        #         results['generated_value'] *= results[
        #             'generated_attention_mask']  # TODO: I doubt if this line is necessary
        #
        # # Run ref policy
        # with torch.no_grad():
        #     ref_policy_forward = self.ref_policy_model.forward_pass(**forward_inputs)
        #     results['generated_ref_logits'] = ref_policy_forward['generated_logits']
        #     results['generated_ref_logprobs'] = ref_policy_forward['generated_logprobs']
        #
        # # Get reward
        # with torch.no_grad():
        #     reward_results = self.reward_model.get_reward(
        #         prompts_input_ids=results['prompts_input_ids'],
        #         prompts_attention_mask=results['prompts_attention_mask'],
        #         generated_input_ids=results['generated_input_ids'],
        #         generated_attention_mask=results['generated_attention_mask'],
        #         generated_texts=results['generated_text'],
        #         metadata=[elem for elem in batch['metadata'] for _ in
        #                   range(self.args['env']['train_num_samples_per_input'])],
        #     )
        #     results.update(reward_results)
        #     self.reward_model.kl_penalize_reward(results)
        #
        # # Get advantages
        # self.compute_advantages(results, self.args['env']['train_num_samples_per_input'])
        #
        # n_results = len(results['generated_input_ids'])
        #
        # loss_totals, loss_policies, loss_values = [], [], []
        # reward_penalizeds, reward_kls, reward_raws = [], [], []
        #
        # # Train
        # # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        #
        # self.policy_model.model.train()
        # self.value_model.model.train()
        # self.value_model.linear.train()
        #
        # for ppo_epoch_idx in range(self.args['train']['n_ppo_epoch_per_rollout']):
        #     self.optimizer.zero_grad()
        #
        #     # get the weight for each sub-batch
        #     mask = results['generated_attention_mask']
        #     all_mask = self.accelerator.gather(mask)
        #     all_mask_weight = all_mask.sum(dim=1).float().mean().item()
        #
        #     for batch_idx in range(0, n_results, self.args['train']['training_batch_size_per_card']):
        #         batch_results = {}
        #
        #         for k, v in results.items():
        #             batch_results[k] = v[batch_idx:batch_idx + self.args['train']['training_batch_size_per_card']]
        #
        #         self.loss(batch_results, all_mask_weight)
        #         # gradient accumulation weight
        #         self.accelerator.backward(batch_results['loss/total'])
        #
        #         # logging
        #         if ppo_epoch_idx == self.args['train']['n_ppo_epoch_per_rollout'] - 1:
        #             loss_total = batch_results['loss/total'].unsqueeze(0)  # (1)
        #             loss_policy = batch_results['loss/policy'].unsqueeze(0)  # (1)
        #             loss_value = batch_results['loss/value'].unsqueeze(0)  # (1)
        #             reward_penalized = torch.mean(
        #                 reduce_sum(batch_results['rewards/penalized'], batch_results['generated_attention_mask'],
        #                            axis=1)).unsqueeze(0)  # (1)
        #             reward_kl = torch.mean(
        #                 reduce_sum(batch_results['rewards/kl'], batch_results['generated_attention_mask'],
        #                            axis=1)).unsqueeze(0)  # (1)
        #             reward_raw = torch.mean(
        #                 reduce_sum(batch_results['rewards/raw'], batch_results['generated_attention_mask'],
        #                            axis=1)).unsqueeze(0)  # (1)
        #
        #             loss_totals.append(loss_total)
        #             loss_policies.append(loss_policy)
        #             loss_values.append(loss_value)
        #             reward_penalizeds.append(reward_penalized)
        #             reward_kls.append(reward_kl)
        #             reward_raws.append(reward_raw)
        #
        #     if self.args['train']['clip_grad']:
        #         self.accelerator.clip_grad_norm_(
        #             chain(self.policy_model.model.parameters(),
        #                   self.policy_model.linear.parameters(),
        #                   self.value_model.model.parameters(),
        #                   self.value_model.linear.parameters()
        #                   ),
        #             self.args['train']['max_grad_norm'])
        #
        #     self.optimizer.step()
        #     self.scheduler.step()
        #
        # loss_total = torch.cat(loss_totals, dim=0)
        # loss_policy = torch.cat(loss_policies, dim=0)
        # loss_value = torch.cat(loss_values, dim=0)
        # reward_penalized = torch.cat(reward_penalizeds, dim=0)
        # reward_kl = torch.cat(reward_kls, dim=0)
        # reward_raw = torch.cat(reward_raws, dim=0)
        #
        # losses_total = self.accelerator.gather(loss_total)  # (num_gpus)
        # losses_policy = self.accelerator.gather(loss_policy)  # (num_gpus)
        #
        # losses_value = self.accelerator.gather(loss_value)  # (num_gpus)
        # rewards_penalized = self.accelerator.gather(reward_penalized)  # (num_gpus)
        # rewards_kl = self.accelerator.gather(reward_kl)  # (num_gpus)
        # rewards_raw = self.accelerator.gather(reward_raw)  # (num_gpus)
        #
        # loss_total = losses_total.mean().item()
        # loss_policy = losses_policy.mean().item()
        #
        # loss_value = losses_value.mean().item()
        # reward_penalized = rewards_penalized.mean().item()
        # reward_kl = rewards_kl.mean().item()
        # reward_raw = rewards_raw.mean().item()
        #
        # # Logging
        # if self.args['logging']['wandb_log'] and self.accelerator.is_main_process:
        #
        #     this_batch_kl = np.mean(reward_kl)
        #
        #     if step % self.args['logging']['log_interval'] == 0:
        #         wandb.log({
        #             'train/step': step,
        #             'train/lr': self.scheduler.get_last_lr()[0],
        #             'train/loss/total': np.mean(loss_total),
        #             'train/loss/policy': np.mean(loss_policy),
        #             'train/loss/value': np.mean(loss_value),
        #             'train/reward/penalized': np.mean(reward_penalized),
        #             'train/reward/KL': this_batch_kl,
        #             'train/reward/raw': np.mean(reward_raw),
        #         })
        #
        #     if this_batch_kl > self.args['train']['kl_threshold']:
        #         self.log_info(f"KL divergence {this_batch_kl} exceeds threshold {self.args['train']['kl_threshold']}")
        #         self.huge_kl_count += 1
        #         if self.huge_kl_count >= 5:
        #             self.should_early_stop = True

    def valid(self, step):
        self.log_info(f'Evaluating [step {step}] ...')

        self.accelerator.wait_for_everyone()

        self.policy_model.model.eval()
        if not self.args['model']['value_model']['policy_value_sharing']:
            self.value_model.model.eval()

        columns = ["step", "inputs", "outputs"]
        wandb_table = None

        n_entries = 0

        with torch.no_grad():
            for i, batch in enumerate(
                    tqdm(self.eval_dataloader) if self.accelerator.is_main_process else self.eval_dataloader):

                results = self.policy_model.sample(
                    prompts_input_ids=batch['prompts_input_ids'],
                    prompts_attention_mask=batch['prompts_attention_mask'],
                    **self.args['model']['policy_model']['eval_generation_kwargs'],
                )

                eval_results = self.reward_model.eval_metrics(
                    prompts_input_ids=results['prompts_input_ids'],
                    prompts_attention_mask=results['prompts_attention_mask'],
                    generated_input_ids=results['generated_input_ids'],
                    generated_attention_mask=results['generated_attention_mask'],
                    generated_texts=results['generated_text'],
                    metadata=batch['metadata'],
                )

                # gather all results
                batch = self.accelerator.gather_for_metrics(batch)
                results = self.accelerator.gather_for_metrics(results)

                for eval_k, eval_v in eval_results.items():
                    eval_results[eval_k] = self.accelerator.gather(
                        torch.tensor(eval_v, device=results['generated_input_ids'].device))

                # initialize wandb table if it does not exist
                if wandb_table is None:
                    columns.extend(list(eval_results.keys()))
                    wandb_table = wandb.Table(columns=columns)

                if self.accelerator.is_main_process:

                    prompt_inputs = self.policy_model.tokenizer.batch_decode(results['prompts_input_ids'],
                                                                             skip_special_tokens=True,
                                                                             clean_up_tokenization_spaces=True)

                    generated_texts = self.policy_model.tokenizer.batch_decode(results['generated_input_ids'],
                                                                               skip_special_tokens=True,
                                                                               clean_up_tokenization_spaces=True)

                    this_data_batch_size = results['prompts_input_ids'].shape[0]
                    this_lens = torch.sum(results['generated_attention_mask'], dim=-1)

                    for batch_i in range(this_data_batch_size):

                        this_entry = [step, prompt_inputs[batch_i], generated_texts[batch_i]]

                        for eval_v in eval_results.values():
                            this_entry.append(eval_v[batch_i].item())

                        wandb_table.add_data(*this_entry)
                        n_entries += 1

        if self.accelerator.is_main_process:

            # do statistics
            n_dev_samples = len(wandb_table.data)

            stats = {'eval/step': step,
                     f'eval_generation/step_{step}': wandb_table}

            value_columns = columns[3:]  # the first three are steps, inputs, outputs
            stats.update(self.reward_model.aggregate_metrics(wandb_table, value_columns))

            if self.args['logging']['wandb_log']:
                wandb.log(stats)

            mean_rewards = stats["eval/rewards"]

            self.log_info(f'Evaluated [step {step}] rewards = {mean_rewards:.4f}')

            prev_best_step = None if len(self.eval_accs) == 0 else max(self.eval_accs, key=self.eval_accs.get)
            self.eval_accs[step] = mean_rewards
            if prev_best_step is None or mean_rewards > self.eval_accs[prev_best_step]:
                if prev_best_step is not None:
                    try:
                        os.remove(f"{self.args['logging']['save_dir']}/ckp_{prev_best_step}.pth")
                    except:
                        self.log_info(f'Cannot remove previous best ckpt!')
                shutil.copy(f"{self.args['logging']['save_dir']}/last.pth",
                            f"{self.args['logging']['save_dir']}/ckp_{step}.pth")
                self.log_info(f'Best ckpt updated to [step {step}]')

                # save best policy again
                self.accelerator.wait_for_everyone()
                policy_model_state_dict = self.accelerator.unwrap_model(self.policy_model.model).state_dict()
                self.accelerator.save(policy_model_state_dict, f"{self.args['logging']['save_dir']}/best_policy.pth")

    def save(self, step):
        # this will overwrite an existing ckpt with the save filename!
        self.accelerator.wait_for_everyone()
        policy_model_state_dict = self.accelerator.unwrap_model(self.policy_model.model).state_dict()
        policy_linear_state_dict = self.accelerator.unwrap_model(self.policy_model.linear).state_dict()
        if not self.args['model']['value_model']['policy_value_sharing']:
            value_model_state_dict = self.accelerator.unwrap_model(self.value_model.model).state_dict()
            value_linear_state_dict = self.accelerator.unwrap_model(self.value_model.linear).state_dict()

        result = {
            'model': policy_model_state_dict,
            'linear': policy_linear_state_dict,
            'step': step,
            'eval_accs': self.eval_accs,
            # 'optimizer': optimizer_state_dict,
        }
        if not self.args['model']['value_model']['policy_value_sharing']:
            result['value_model'] = value_model_state_dict
            result['value_linear'] = value_linear_state_dict
        self.accelerator.wait_for_everyone()
        self.accelerator.save(result, f"{self.args['logging']['save_dir']}/last.pth")
        self.log_info(f'[step {step}] model checkpoint saved')

    def extract_nouns(self, text):
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        nouns = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith('NN')]
        return nouns
    def init(self):
        metrics = {}
        with open(self.args.metrics, "r") as file:
            lines = file.readlines()

        for line in lines:
            parts = line.strip().split('=')
            if len(parts) == 2:
                variable_name = parts[0].strip()
                variable_value = eval(parts[1].strip())
                metrics[variable_name] = variable_value

        return metrics

    def check_synonyms_word(self, word1, word2, similarity_score):
        token1 = nlp(word1)
        token2 = nlp(word2)
        similarity = token1.similarity(token2)
        return similarity > similarity_score
    def chair(self, id, response):
        metrics = self.init()
        association = json.load(open(self.args.word_association, 'r', encoding='utf-8'))
        hallucination_words = []
        for word1 in association.keys():
            hallucination_words.append(word1)
            for word2 in association[word1]:
                hallucination_words.append(word2)

        global_safe_words = []
        with open(self.args.safe_words, 'r', encoding='utf-8') as safe_file:
            for line in safe_file:
                line = line.split('\n')[0]
                global_safe_words.append(line)

        dimension = {'g': False, 'de': False, 'da': False, 'dr': False}
        if self.args.evaluation_type == 'a':
            for key in dimension.keys():
                dimension[key] = True
        elif self.args.evaluation_type == 'g':
            dimension['g'] = True
        elif self.args.evaluation_type == 'd':
            dimension['de'] = True
            dimension['da'] = True
            dimension['dr'] = True
        else:
            dimension[self.args.evaluation_type] = True

        inference_data = json.load(open(self.args.inference_data, 'r', encoding='utf-8'))
        ground_truth = json.load(open(self.args.annotation, 'r', encoding='utf-8'))



        if ground_truth[id - 1]['type'] == 'generative':
            nouns = self.extract_nouns(response)
            after_process_nouns = []
            for noun in nouns:
                if noun in hallucination_words:
                    after_process_nouns.append(noun)

            safe_words = []
            safe_list = []
            for idx, word in enumerate(ground_truth[id - 1]['truth']):
                safe_words += association[word]
                safe_list += [idx] * len(association[word])

            ha_words = []
            ha_list = []
            for idx, word in enumerate(ground_truth[id - 1]['hallu']):
                ha_words += association[word]
                ha_list += [idx] * len(association[word])

            safe_words += ground_truth[id - 1]['truth']
            safe_len = len(ground_truth[id - 1]['truth'])
            safe_list += [0] * safe_len
            safe_flag_list = [0] * len(after_process_nouns)

            ha_words += ground_truth[id - 1]['hallu']
            ha_len = len(ground_truth[id - 1]['hallu'])
            ha_list += [0] * ha_len

            for idx, noun in enumerate(after_process_nouns):
                if noun in global_safe_words:
                    continue

                if noun in safe_words:
                    for j in range(len(safe_words)):
                        if noun == safe_words[j]:
                            if j < (len(safe_list) - safe_len):
                                safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                            else:
                                safe_list[j] = 1
                            break
                    continue

                if noun in ha_words:
                    for j in range(len(ha_words)):
                        if noun == ha_words[j]:
                            if j < (len(ha_list) - ha_len):
                                ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                            else:
                                ha_list[j] = 1
                            break

                for j, check_word in enumerate(ha_words):
                    if self.check_synonyms_word(noun, check_word, self.args.similarity_score):
                        if j < (len(ha_list) - ha_len):
                            ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                        else:
                            ha_list[j] = 1
                        break

                flag = False
                for j, check_word in enumerate(safe_words):
                    if self.check_synonyms_word(noun, check_word, self.args.similarity_score):
                        flag = True
                        if j < (len(safe_list) - safe_len):
                            safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                        else:
                            safe_list[j] = 1
                        break
                if flag == True:
                    continue

                safe_flag_list[idx] = 1

            metrics['chair_score'] += sum(safe_flag_list)
            metrics['chair_num'] += len(safe_flag_list)
            metrics['safe_cover_score'] += sum(safe_list[-safe_len:])
            metrics['safe_cover_num'] += len(safe_list[-safe_len:])
            metrics['hallu_cover_score'] += sum(ha_list[-ha_len:])
            metrics['hallu_cover_num'] += len(ha_list[-ha_len:])
            if sum(safe_flag_list) == 0:
                metrics['non_hallu_score'] += 1
            metrics['non_hallu_num'] += 1

        # else:
        #     metrics['qa_correct_num'] += 1
        #     if ground_truth[id - 1]['type'] == 'discriminative-attribute-state':
        #         metrics['as_qa_correct_num'] += 1
        #     elif ground_truth[id - 1]['type'] == 'discriminative-attribute-number':
        #         metrics['an_qa_correct_num'] += 1
        #     elif ground_truth[id - 1]['type'] == 'discriminative-attribute-action':
        #         metrics['aa_qa_correct_num'] += 1
        #     elif ground_truth[id - 1]['type'] == 'discriminative-hallucination':
        #         metrics['ha_qa_correct_num'] += 1
        #     else:
        #         metrics['asso_qa_correct_num'] += 1
        #
        #     truth = ground_truth[id - 1]['truth']
        #     response = inference_data[i]['response']
        #     if truth == 'yes':
        #         if response == 'Yes':
        #             metrics['qa_correct_score'] += 1
        #             if ground_truth[id - 1]['type'] == 'discriminative-attribute-state':
        #                 metrics['as_qa_correct_score'] += 1
        #             elif ground_truth[id - 1]['type'] == 'discriminative-attribute-number':
        #                 metrics['an_qa_correct_score'] += 1
        #             elif ground_truth[id - 1]['type'] == 'discriminative-attribute-action':
        #                 metrics['aa_qa_correct_score'] += 1
        #             elif ground_truth[id - 1]['type'] == 'discriminative-hallucination':
        #                 metrics['ha_qa_correct_score'] += 1
        #             else:
        #                 metrics['asso_qa_correct_score'] += 1
        #     else:
        #         metrics['qa_no_num'] += 1
        #         if ground_truth[id - 1]['type'] == 'discriminative-attribute-state':
        #             metrics['as_qa_no_num'] += 1
        #         elif ground_truth[id - 1]['type'] == 'discriminative-attribute-number':
        #             metrics['an_qa_no_num'] += 1
        #         elif ground_truth[id - 1]['type'] == 'discriminative-attribute-action':
        #             metrics['aa_qa_no_num'] += 1
        #         elif ground_truth[id - 1]['type'] == 'discriminative-hallucination':
        #             metrics['ha_qa_no_num'] += 1
        #         else:
        #             metrics['asso_qa_no_num'] += 1
        #
        #         if response == 'No':
        #             metrics['qa_correct_score'] += 1
        #             metrics['qa_no_score'] += 1
        #             if ground_truth[id - 1]['type'] == 'discriminative-attribute-state':
        #                 metrics['as_qa_correct_score'] += 1
        #                 metrics['as_qa_no_score'] += 1
        #             elif ground_truth[id - 1]['type'] == 'discriminative-attribute-number':
        #                 metrics['an_qa_correct_score'] += 1
        #                 metrics['an_qa_no_score'] += 1
        #             elif ground_truth[id - 1]['type'] == 'discriminative-attribute-action':
        #                 metrics['aa_qa_correct_score'] += 1
        #                 metrics['aa_qa_no_score'] += 1
        #             elif ground_truth[id - 1]['type'] == 'discriminative-hallucination':
        #                 metrics['ha_qa_correct_score'] += 1
        #                 metrics['ha_qa_no_score'] += 1
        #             else:
        #                 metrics['asso_qa_correct_score'] += 1
        #                 metrics['asso_qa_no_score'] += 1
        #
        #     if response == 'No':
        #         metrics['qa_ans_no_num'] += 1
        #         if ground_truth[id - 1]['type'] == 'discriminative-attribute-state':
        #             metrics['as_qa_ans_no_num'] += 1
        #         elif ground_truth[id - 1]['type'] == 'discriminative-attribute-number':
        #             metrics['an_qa_ans_no_num'] += 1
        #         elif ground_truth[id - 1]['type'] == 'discriminative-attribute-action':
        #             metrics['aa_qa_ans_no_num'] += 1
        #         elif ground_truth[id - 1]['type'] == 'discriminative-hallucination':
        #             metrics['ha_qa_ans_no_num'] += 1
        #         else:
        #             metrics['asso_qa_ans_no_num'] += 1
        #         if truth == 'no':
        #             metrics['qa_ans_no_score'] += 1
        #             if ground_truth[id - 1]['type'] == 'discriminative-attribute-state':
        #                 metrics['as_qa_ans_no_score'] += 1
        #             elif ground_truth[id - 1]['type'] == 'discriminative-attribute-number':
        #                 metrics['an_qa_ans_no_score'] += 1
        #             elif ground_truth[id - 1]['type'] == 'discriminative-attribute-action':
        #                 metrics['aa_qa_ans_no_score'] += 1
        #             elif ground_truth[id - 1]['type'] == 'discriminative-hallucination':
        #                 metrics['ha_qa_ans_no_score'] += 1
        #             else:
        #                 metrics['asso_qa_ans_no_score'] += 1

        if dimension['g']:
            CHAIR = round(metrics['chair_score'] / metrics['chair_num'] * 100, 1)
            Cover = round(metrics['safe_cover_score'] / metrics['safe_cover_num'] * 100, 1)
            Ha = round(metrics['hallu_cover_score'] / metrics['hallu_cover_num'] * 100, 1)
            Ha_p = round(100 - metrics['non_hallu_score'] / metrics['non_hallu_num'] * 100, 1)
            # print("Generative Task:")
            # print("CHAIR:\t\t", CHAIR)
            # print("Cover:\t\t", Cover)
            # print("Hal:\t\t", Ha_p)
            # print("Cog:\t\t", Ha, "\n")
        return CHAIR
        #
        # if dimension['de'] and dimension['da'] and dimension['dr']:
        #     Accuracy = round(metrics['qa_correct_score'] / metrics['qa_correct_num'] * 100, 1)
        #     Precision = round(metrics['qa_ans_no_score'] / metrics['qa_ans_no_num'] * 100, 1)
        #     Recall = round(metrics['qa_no_score'] / metrics['qa_no_num'] * 100, 1)
        #     F1 = round(2 * (Precision / 100) * (Recall / 100) / ((Precision / 100) + (Recall / 100) + 0.0001) * 100, 1)
        #     print("Descriminative Task:")
        #     print("Accuracy:\t", Accuracy)
        #     print("Precision:\t", Precision)
        #     print("Recall:\t\t", Recall)
        #     print("F1:\t\t", F1, "\n")
        #
        # if dimension['de']:
        #     hallucination_Accuracy = round(metrics['ha_qa_correct_score'] / metrics['ha_qa_correct_num'] * 100, 1)
        #     hallucination_Precision = round(metrics['ha_qa_ans_no_score'] / metrics['ha_qa_ans_no_num'] * 100, 1)
        #     hallucination_Recall = round(metrics['ha_qa_no_score'] / metrics['ha_qa_no_num'] * 100, 1)
        #     hallucination_F1 = round(2 * (hallucination_Precision / 100) * (hallucination_Recall / 100) / (
        #                 (hallucination_Precision / 100) + (hallucination_Recall / 100) + 0.001) * 100, 1)
        #     print("Exsitence:")
        #     print("Accuracy:\t", hallucination_Accuracy)
        #     print("Precision:\t", hallucination_Precision)
        #     print("Recall:\t\t", hallucination_Recall)
        #     print("F1:\t\t", hallucination_F1, "\n")
        #
        # if dimension['da']:
        #     attr_Accuracy = round(
        #         (metrics['as_qa_correct_score'] + metrics['an_qa_correct_score'] + metrics['aa_qa_correct_score']) / (
        #                     metrics['as_qa_correct_num'] + metrics['an_qa_correct_num'] + metrics[
        #                 'aa_qa_correct_num']) * 100, 1)
        #     attr_Precision = round(
        #         (metrics['as_qa_ans_no_score'] + metrics['an_qa_ans_no_score'] + metrics['aa_qa_ans_no_score']) / (
        #                     metrics['as_qa_ans_no_num'] + metrics['an_qa_ans_no_num'] + metrics[
        #                 'aa_qa_ans_no_num']) * 100, 1)
        #     attr_Recall = round((metrics['as_qa_no_score'] + metrics['an_qa_no_score'] + metrics['aa_qa_no_score']) / (
        #                 metrics['as_qa_no_num'] + metrics['an_qa_no_num'] + metrics['aa_qa_no_num']) * 100, 1)
        #     attr_F1 = round(2 * (attr_Precision / 100) * (attr_Recall / 100) / (
        #                 (attr_Precision / 100) + (attr_Recall / 100) + 0.0001) * 100, 1)
        #     state_Accuracy = round(metrics['as_qa_correct_score'] / metrics['as_qa_correct_num'] * 100, 1)
        #     state_Precision = round(metrics['as_qa_ans_no_score'] / metrics['as_qa_ans_no_num'] * 100, 1)
        #     state_Recall = round(metrics['as_qa_no_score'] / metrics['as_qa_no_num'] * 100, 1)
        #     state_F1 = round(2 * (state_Precision / 100) * (state_Recall / 100) / (
        #                 (state_Precision / 100) + (state_Recall / 100) + 0.0001) * 100, 1)
        #     number_Accuracy = round(metrics['an_qa_correct_score'] / metrics['an_qa_correct_num'] * 100, 1)
        #     number_Precision = round(metrics['an_qa_ans_no_score'] / metrics['an_qa_ans_no_num'] * 100, 1)
        #     number_Recall = round(metrics['an_qa_no_score'] / metrics['an_qa_no_num'] * 100, 1)
        #     number_F1 = round(2 * (number_Precision / 100) * (number_Recall / 100) / (
        #                 (number_Precision / 100) + (number_Recall / 100) + 0.0001) * 100, 1)
        #     action_Accuracy = round(metrics['aa_qa_correct_score'] / metrics['aa_qa_correct_num'] * 100, 1)
        #     action_Precision = round(metrics['aa_qa_ans_no_score'] / metrics['aa_qa_ans_no_num'] * 100, 1)
        #     action_Recall = round(metrics['aa_qa_no_score'] / metrics['aa_qa_no_num'] * 100, 1)
        #     action_F1 = round(2 * (action_Precision / 100) * (action_Recall / 100) / (
        #                 (action_Precision / 100) + (action_Recall / 100) + 0.0001) * 100, 1)
        #     print("Attribute:")
        #     print("Accuracy:\t", attr_Accuracy)
        #     print("Precision:\t", attr_Precision)
        #     print("Recall:\t\t", attr_Recall)
        #     print("F1:\t\t", attr_F1, "\n")
        #     print("State:")
        #     print("Accuracy:\t", state_Accuracy)
        #     print("Precision:\t", state_Precision)
        #     print("Recall:\t\t", state_Recall)
        #     print("F1:\t\t", state_F1, "\n")
        #     print("Number:")
        #     print("Accuracy:\t", number_Accuracy)
        #     print("Precision:\t", number_Precision)
        #     print("Recall:\t\t", number_Recall)
        #     print("F1:\t\t", number_F1, "\n")
        #     print("Action:")
        #     print("Accuracy:\t", action_Accuracy)
        #     print("Precision:\t", action_Precision)
        #     print("Recall:\t\t", action_Recall)
        #     print("F1:\t\t", action_F1, "\n")
        #
        # if dimension['dr']:
        #     relation_Accuracy = round(metrics['asso_qa_correct_score'] / metrics['asso_qa_correct_num'] * 100, 1)
        #     relation_Precision = round(metrics['asso_qa_ans_no_score'] / metrics['asso_qa_ans_no_num'] * 100, 1)
        #     relation_Recall = round(metrics['asso_qa_no_score'] / metrics['asso_qa_no_num'] * 100, 1)
        #     relation_F1 = round(2 * (relation_Precision / 100) * (relation_Recall / 100) / (
        #                 (relation_Precision / 100) + (relation_Recall / 100) + 0.0001) * 100, 1)
        #     print("Relation:")
        #     print("Accuracy:\t", relation_Accuracy)
        #     print("Precision:\t", relation_Precision)
        #     print("Recall:\t\t", relation_Recall)
        #     print("F1:\t\t", relation_F1)

