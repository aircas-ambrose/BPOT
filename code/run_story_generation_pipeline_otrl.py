# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import logging
import argparse
import json
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, T5ForConditionalGeneration, BartForConditionalGeneration
from utils import *
from optimization import *
from pathlib import Path
import re
import math
from collections import Counter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

PYTORCH_PRETRAINED_ROBERTA_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_ROBERTA_CACHE',
                                                  Path.home() / '.pytorch_pretrained_roberta'))


def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
        This is taken directly from modeling_bart.py
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cnt(loss):
    cnter = torch.Tensor([torch.count_nonzero(l) for l in loss])
    cnter.require_grad = False
    return cnter.to(device)


def fine_grained_for_ot(args, loss_bak, output_ids, reward, output_ids_s_index, cross_ot_allocate_bak, use_pred_mask):
    """fintune the scaling in reinforcement learning"""
    loss_sclaing_matrix = torch.ones_like(loss_bak, dtype=torch.float32)
    storyline_prompt_index = torch.tensor(np.argwhere(output_ids.cpu().numpy() >= 50265))
    batch_prompt_index = storyline_prompt_index[:, :1].reshape(-1)
    batch_prompt_num = [torch.sum(batch_prompt_index == torch.tensor(i, dtype=torch.int64)).item() for i in range(args.train_batch_size)]
    batch_prompt_position = []
    batch_sentence_position = []
    for i in range(args.train_batch_size):
        prompt_start = 1
        sentence_start = 1
        cur_example_num = 0 if i == 0 else sum(batch_prompt_num[:i])
        prompt_position = []
        sentence_position = []
        for j in range(batch_prompt_num[i]):
            assert batch_prompt_num[i] == len(output_ids_s_index[i])
            prompt_end = storyline_prompt_index[cur_example_num + j][1].item() + 1
            sentence_end = output_ids_s_index[i][j].item() + 1
            prompt_position.append((prompt_start, prompt_end))
            sentence_position.append((sentence_start, sentence_end))
            prompt_start = prompt_end - 1
            sentence_start = sentence_end
        batch_prompt_position.append(prompt_position)
        batch_sentence_position.append(sentence_position)

    batch_reward_loss = []
    for i in range(args.train_batch_size):
        prompt_position = batch_prompt_position[i]
        sentence_position = batch_sentence_position[i]
        assert len(prompt_position) == len(sentence_position)
        reward_loss = []
        for j in range(len(sentence_position)):
            reward_loss.append(torch.sum(reward[i, sentence_position[j][0]:sentence_position[j][1]]) /
                               torch.tensor(sentence_position[j][1] - sentence_position[j][0], dtype=torch.float32))
        batch_reward_loss.append(reward_loss)
    batch_reward_loss = torch.tensor(batch_reward_loss).to(device)
    assert batch_reward_loss.size(1) == cross_ot_allocate_bak.size(2)
    allocate_matrix = batch_reward_loss.size(1) * torch.bmm(cross_ot_allocate_bak, batch_reward_loss.unsqueeze(-1)).squeeze()
    allocate_matrix = allocate_matrix / allocate_matrix.mean(dim=1, keepdim=True)
    for i in range(args.train_batch_size):
        use_pred = use_pred_mask[i]
        if use_pred == 0:
            continue
        prompt_position = batch_prompt_position[i]
        for j in range(len(prompt_position)):
            loss_sclaing_matrix[i, prompt_position[j][0]:prompt_position[j][1] - 1] = allocate_matrix[i][j]
            if j != 0:
                loss_sclaing_matrix[i, prompt_position[j][0]] += allocate_matrix[i][j-1]
            if j == len(prompt_position) -1:
                loss_sclaing_matrix[i, prompt_position[j][1] - 1] = allocate_matrix[i][j]
    loss_sclaing_matrix.requires_grad = False
    return loss_sclaing_matrix


def find_argument_interval_for_event(token_result, event_index, sep_token=25606, storyline_flag=False, own_full_stop=False):
    event_index = event_index.cpu().numpy().tolist()
    sep_token_id = sep_token
    batch_size = len(event_index)
    batch_event_position = []
    for i in range(batch_size):
        special_token_position = event_index[i]
        if not storyline_flag:
            assert len(special_token_position) == 5
        else:
            assert len(special_token_position) == 6
        story_events_position = []
        start_position = 1
        for j in range(len(special_token_position)):
            end_position = special_token_position[j] if not own_full_stop else special_token_position[j] + 1
            event_position = (start_position, end_position)
            start_position = end_position + 1
            story_events_position.append(event_position)
        if not storyline_flag:
            assert len(story_events_position) == 5
        else:
            assert len(story_events_position) == 6
        batch_event_position.append(story_events_position)
    return batch_event_position

def get_event_representation(batch_event_position, last_hidden_states, device, event_transform=None):
    """convert the word representation to event representation"""
    batch_size = last_hidden_states.shape[0]
    event_num = len(batch_event_position[0])
    hidden_dim = last_hidden_states.shape[-1]
    if event_transform != None:
        last_hidden_states = event_transform(last_hidden_states)
    batch_event_representation = torch.zeros((batch_size, event_num, hidden_dim))
    for i in range(batch_size):
        event_position = batch_event_position[i]
        for j in range(len(event_position)):
            event_sample_position = event_position[j]
            batch_event_representation[i, j, :] = torch.mean(
                last_hidden_states[i, event_sample_position[0]: event_sample_position[1], :], dim=0).to(device)
    return batch_event_representation

def batch_gaussian_cost_matrix_torch(x, y, gamma):
    norm_x = x.div(torch.norm(x, p=2, dim=2, keepdim=True) + 1e-12)
    norm_y = y.div(torch.norm(y, p=2, dim=2, keepdim=True) + 1e-12)
    norm_x_square = torch.square(norm_x).sum(dim=2, keepdim=True)
    norm_y_square = torch.square(norm_y).sum(dim=2, keepdim=True).reshape(y.size(0), 1, -1)
    norm_xy = torch.bmm(norm_x, torch.transpose(norm_y, 1, 2))
    xy_distance = norm_x_square + norm_y_square - 2 * norm_xy
    gaussian_distance = 1 - torch.exp(-xy_distance / (2 * gamma**2))
    return gaussian_distance

def batch_cost_matrix_torch(x, y):
    """基于余弦相似度计算两个对应沙堆的cost矩阵"""
    norm_x = x.div(torch.norm(x, p=2, dim=2, keepdim=True) + 1e-12)
    norm_y = y.div(torch.norm(y, p=2, dim=2, keepdim=True) + 1e-12)
    cos_distance = torch.bmm(norm_x, torch.transpose(norm_y, 1, 2))
    cos_distance = 1 - cos_distance
    return cos_distance


def batch_transport_matrix_torch(cost_matrix, batch_size, n, m, device, beta=0.1, iteration=5):
    """根据cost matrix迭代得到transport matrix"""
    sigma = torch.ones(batch_size, int(m), 1).to(device) / float(m)
    T = torch.ones(batch_size, n, m).to(device)
    A = torch.exp(-cost_matrix/beta).float().to(device)
    for t in range(iteration):
        Q = A * T
        for k in range(1):
            delta = 1 / (n * torch.bmm(Q, sigma))
            a = torch.bmm(torch.transpose(Q, 1, 2), delta)
            sigma = 1 / (float(m) * a)
        T = delta * Q * sigma.transpose(2, 1)
    return T

def batch_trace(input_matrix, device):
    batch_size = input_matrix.shape[0]
    group_size = input_matrix.shape[1]
    mask_matrix = torch.eye(group_size).to(device).unsqueeze(0).repeat(batch_size, 1, 1)
    mask_input_matrix = mask_matrix * input_matrix
    return torch.sum(torch.sum(mask_input_matrix, -1), -1).unsqueeze(1)


def compute_regularization_loss(transport_matrix):
    batch_size = transport_matrix.size(0)
    n, m = transport_matrix.size(1), transport_matrix.size(2)
    assert n == m
    logits = m * transport_matrix.reshape(batch_size * n, m)
    labels = [i for i in range(m)] * batch_size
    labels = torch.tensor(labels, dtype=torch.long).reshape(batch_size * n, 1).to(device)
    regularization_loss = nn.CrossEntropyLoss(reduction='none')
    reg_loss = regularization_loss(logits, labels.squeeze()).reshape(batch_size, n)
    reg_loss = torch.mean(reg_loss.sum(dim=1))
    return reg_loss


def ipot_distance_loss(x, y, device, beta=0.1, gamma=1, iteration=5):
    """完成整个ipot loss的计算步骤"""
    batch_size = x.shape[0]
    n = x.shape[1]
    m = y.shape[1]
    cost_matrix = batch_gaussian_cost_matrix_torch(x, y, gamma)
    cost_matrix = cost_matrix.float().to(device)
    transport_matrix = batch_transport_matrix_torch(cost_matrix, batch_size, n, m, device,
                                                    beta=beta, iteration=iteration)
    temp = torch.bmm(torch.transpose(cost_matrix, 1, 2), transport_matrix)
    ot_loss = batch_trace(temp, device)
    return ot_loss, transport_matrix, None


def convert_scalar_index_to_tuple(scalar_index):
    scalar_index = scalar_index.cpu().numpy()
    batch_tuple_index = []
    batch_size = scalar_index.shape[0]
    event_num = scalar_index.shape[1]
    for i in range(batch_size):
        tuple_index = []
        event_tuple_start = 1
        event_tuple_end = scalar_index[i][0] + 1
        tuple_index.append((event_tuple_start, event_tuple_end))
        event_tuple_start = event_tuple_end
        for j in range(event_num - 1):
            event_tuple_end = scalar_index[i][j+1] + 1
            tuple_index.append((event_tuple_start, event_tuple_end))
            event_tuple_start = event_tuple_end
        batch_tuple_index.append(tuple_index)
    return batch_tuple_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="pre-trained model selected in the list: "
                             "allenai/unifiedqa-t5-{small, base, large ...} ")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--file_suffix",
                        default=None,
                        type=str,
                        required=True,
                        help="unique identifier for data file")
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=32,  # 8 * 8 * 5
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--gen_storyline_len",
                        default=128,  # 8 * 8 * 5
                        type=int,
                        help="The maximum length for generated storyline.")
    parser.add_argument("--sub_sample",
                        default=0, # 8 * 8 * 5
                        type=int,
                        help="0 means full data; otherwise, use K random sample for training")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--finetune",
                        action='store_true',
                        help="Whether to finetune LM.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--pos_weight',
                        type=int,
                        default=1,
                        help="positive weight on label 1")
    parser.add_argument("--load_model",
                        type=str,
                        help="pretrained model dir",
                        default="")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--mlp_hid_size",
                        default=64,
                        type=int,
                        help="hid dimension for MLP layer.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--cuda',
                        type=str,
                        default="",
                        help="cuda index")
    parser.add_argument('--save_model',
                        action='store_true',
                        help="save best or not")
    parser.add_argument('--device_num',
                        type=str,
                        default="0",
                        help="cuda device number")
    parser.add_argument('--input_event_num',
                        type=int,
                        default=1,
                        help="input event number")
    parser.add_argument('--mu',
                        type=int,
                        default=0,
                        help="exponential decay factor")
    parser.add_argument('--mu_switch',
                        action='store_true',
                        help='decide whether using the mix training strategy combing two stage and rl')
    parser.add_argument('--num_beams',
                        type=int,
                        default=4,
                        help="number of beams; default 4 (bart)")
    # change here to load the intermediate checkpoint
    parser.add_argument('--intermediate_ckpt',
                        type=str,
                        default='',
                        help='the path to load the intermediate checkpoint')
    parser.add_argument('--alpha',
                        type=float,
                        default=1,
                        help='the coefficiency for the storyline ot loss')
    parser.add_argument('--beta',
                        type=float,
                        default=0.1,
                        help='the coefficiency for the story ot loss')
    parser.add_argument('--gamma',
                        type=float,
                        default=1,
                        help='the coefficiency for the similarity')
    parser.add_argument('--iteration',
                        type=int,
                        default=5,
                        help='the count of ot iteration')
    parser.add_argument('--mu_epoch',
                        type=int,
                        default=0,
                        help='if use mu switch, decide how much epoch')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.cuda.set_device(torch.device('cuda:%s' % args.device_num))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        logger.info(torch.cuda.current_device())
        n_gpu = len(args.device_num.split(','))
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    logger.info("ot configuration | alpha : {} | beta : {} | gama : {} |iteration : {} |".format(args.alpha,
                                                                                      args.beta, args.gamma,
                                                                                                 args.iteration))
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_relation = True if "with_rel" in args.output_dir else False
    no_struct = True if "no_struct" in args.output_dir else False
    use_story_input = True if "story_input" in args.output_dir else False
    template = True if "event_script" in args.output_dir else False
    mu_switch = True if args.mu_switch else False
    double = True if 'double' in args.output_dir else False
    noise = True if 'noise' in args.output_dir else False
    if not use_relation:
        logger.info('not using the temporal relation!')
    if no_struct:
        logger.info('not using the structured storyline!')
    if not use_story_input:
        logger.info('not using the story input!')
    if not template:
        logger.info("not using the event script as the input!")
    if mu_switch:
        logger.info("using the mix training strategy")
    if double:
        logger.info("load pretrained model both in storyline model and storymodel")
    if noise:
        logger.info("add some noise to pretrained model to avoid overfitting")

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.load_model:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()
    logger.info("current task is " + str(task_name))

    # construct model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if no_struct:
        num_added_toks = tokenizer.add_tokens(['<eoe>'])
    else:
        num_added_toks = tokenizer.add_tokens(['<eoe>', '<before>', '<after>', '<vague>'])

    logger.info('We have added %s tokens' % num_added_toks)

    if "t5" in args.model:
        model = T5ForConditionalGeneration.from_pretrained(args.model)
        story_model = T5ForConditionalGeneration.from_pretrained(args.model)
    if "bart" in args.model:
        model = BartForConditionalGeneration.from_pretrained(args.model)
        story_model = BartForConditionalGeneration.from_pretrained(args.model)

    model.resize_token_embeddings(len(tokenizer))
    story_model.resize_token_embeddings(len(tokenizer))
    if args.load_model:
        logger.info(args.load_model)
        model_state_dict = torch.load(args.load_model)
        if len(model_state_dict.keys()) <= 2:
            logger.info('using the pretrained model trained by temporalbart dataset!')
            model_state_dict = model_state_dict['model']
            tmp_storyline_state_dict = {}
            tmp_story_state_dict = {}
            for key, value in model_state_dict.items():
                if not noise:
                    tmp_storyline_state_dict[key.split('mlm.')[1]] = value
                    tmp_story_state_dict[key.split('mlm.')[1]] = value
                else:
                    tmp_storyline_state_dict[key.split('mlm.')[1]] = value + 0.10 * (torch.rand(value.size()) - 0.5) * torch.std(value)
                    tmp_story_state_dict[key.split('mlm.')[1]] = value + 0.15 * (torch.rand(value.size()) - 0.5) * torch.std(value)
            if double:
                story_model.load_state_dict(tmp_story_state_dict)
                model.load_state_dict(tmp_storyline_state_dict)
            else:
                model.load_state_dict(tmp_storyline_state_dict)
    else:
        logger.info('not using the pretrained storyline model!')

    if args.intermediate_ckpt:
        logger.info(args.intermediate_ckpt)
        model_state_dict = torch.load(os.path.join(args.intermediate_ckpt, 'pytorch_model.bin'))
        model_s_state_dict = torch.load(os.path.join(args.intermediate_ckpt, 'pytorch_model_s.bin'))
        model.load_state_dict(model_state_dict)
        story_model.load_state_dict(model_s_state_dict)
    else:
        logger.info('not using the pretrained model!')
    hidden_dim = 768
    storyline_event_transform, story_event_transform = None, None
    event_transform = None
    if event_transform != None:
        logger.info('using one event transform')
        event_transform.to(device)
    if storyline_event_transform != None and story_event_transform != None:
        logger.info('using event transform')
        storyline_event_transform.to(device)
        story_event_transform.to(device)
    model.to(device)
    story_model.to(device)
    if args.do_train:
        with open("%s%s%s" % (args.data_dir, "train", args.file_suffix)) as infile:
            train_data = json.load(infile)

        if args.sub_sample > 0:
            random.Random(args.seed).shuffle(train_data)
            train_data = train_data[:args.sub_sample]

        if 'wp' in args.file_suffix:
            train_features = convert_to_wp_features(train_data, input_event_num=args.input_event_num,
                                                    use_relation=use_relation, no_struct=no_struct, is_eval=True,
                                                    use_story_input=use_story_input)
        elif template:
            train_features = convert_event_script_to_input_update(train_data, input_event_num=args.input_event_num,
                                                                  use_relation=use_relation, no_struct=no_struct,
                                                                  is_eval=True, use_story_input=use_story_input,
                                                                  is_rl_sentence=True, is_otrl_ot=False)
        else:
            train_features = convert_graph_to_features_update(train_data, input_event_num=args.input_event_num,
                                                       use_relation=use_relation, no_struct=no_struct, is_eval=True,
                                                       use_story_input=use_story_input, is_sentence_rl=True,
                                                       is_otrl_ot=False)
        num_train_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        all_inputs = select_field(train_features, 'inputs')
        encoded_inputs = tokenizer(all_inputs, padding=True, truncation=True, return_tensors='pt')

        all_input_ids = encoded_inputs['input_ids']
        all_input_mask = encoded_inputs['attention_mask']

        all_labels = select_field(train_features, 'labels')
        encoded_outputs, all_output_ids_index = convert_feature_to_index(all_labels, tokenizer)

        all_output_ids = encoded_outputs['input_ids']
        all_output_mask = encoded_outputs['attention_mask']

        all_split_story = select_field(train_features, 'split_story')
        encoded_outputs_stories, all_output_ids_s_index = encode_all_stories_update(all_split_story, tokenizer)

        all_output_ids_stories = encoded_outputs_stories['input_ids']
        all_output_mask_stories = encoded_outputs_stories['attention_mask']

        if use_story_input:
            all_story_inputs = select_field(train_features, 'story_input')

        all_key_indices = torch.tensor(list(range(len(all_labels))), dtype=torch.long)

        logger.info("input_id_size: {}, input_mask_size: {},"
                    "instance_key_size: {}, output_id_size: {}, output_mask_size: {}, "
                    "output_index_size: {}, output_id_s_size: {}, output_mask_s_size: {}, "
                    "output_id_s_index_size: {}".format(
            all_input_ids.size(), all_input_mask.size(), all_key_indices.size(),
            all_output_ids.size(), all_output_mask.size(), all_output_ids_index.size(),
            all_output_ids_stories.size(), all_output_mask_stories.size(), all_output_ids_s_index.size()))

        train_data = TensorDataset(all_input_ids, all_input_mask, all_key_indices,
                                   all_output_ids, all_output_mask, all_output_ids_index, all_output_ids_stories,
                                   all_output_mask_stories, all_output_ids_s_index)

        dev_file_suffix = args.file_suffix.split('_new_full_stop')[1]
        with open("%s%s%s" % (args.data_dir, "dev", dev_file_suffix)) as infile:
            eval_data = json.load(infile)

        if 'wp' in args.file_suffix:
            eval_features = convert_to_wp_features(eval_data, input_event_num=args.input_event_num,
                                                    use_relation=use_relation, no_struct=no_struct, is_eval=True,
                                                    use_story_input=use_story_input)
        elif template:
            eval_features = convert_event_script_to_input(eval_data, input_event_num=0, is_eval=True,
                                                          use_relation=use_relation, no_struct=no_struct,
                                                          use_story_input=use_story_input)

        else:
            eval_features = convert_graph_to_features(eval_data, input_event_num=0, is_eval=True,
                                                          use_relation=use_relation, no_struct=no_struct,
                                                          use_story_input=use_story_input)

        del train_features

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        model.train()
        story_model.train()

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        model_to_save_s = model.module if hasattr(story_model, 'module') else story_model
        output_model_file_s = os.path.join(args.output_dir, "pytorch_model_s.bin")
        output_perf_file = os.path.join(args.output_dir, "dev_perf.txt")

        if storyline_event_transform != None and storyline_event_transform != None:
            param_optimizer = list(model.named_parameters()) + list(story_model.named_parameters()) + \
                              list(storyline_event_transform.named_parameters()) + \
                              list(story_event_transform.named_parameters())
        elif event_transform != None:
            param_optimizer = list(event_transform.named_parameters()) + list(model.named_parameters()) + \
                                                                              list(story_model.named_parameters())
        else:
            param_optimizer = list(model.named_parameters()) + list(story_model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = num_train_steps

        if args.fp16:
            try:
                from apex.optimizers import FusedAdam
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False)
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=t_total)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        best_eval_perplexity = float('inf')
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss, tr_acc_start, tr_acc_end, reg_loss_record = 0.0, 0.0, 0.0, 0.0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_masks, instance_indices, output_ids, output_masks, output_index, \
                output_ids_s, output_masks_s, output_ids_s_index = batch
                if 't5' in args.model:
                    loss, _, _, _ = model(input_ids, attention_mask=input_masks,
                                      labels=output_ids, decoder_attention_mask=output_masks)
                    loss_s = 0.0
                else:
                    decoder_output_ids = shift_tokens_right(output_ids, tokenizer.pad_token_id)
                    output = model(input_ids, attention_mask=input_masks,
                                        decoder_input_ids=decoder_output_ids, output_hidden_states=True)

                    output_id_event_index = find_argument_interval_for_event(output_ids, output_index,
                                                                             storyline_flag=False)
                    output_ids_event_representation = get_event_representation(output_id_event_index,
                                                                               output['decoder_hidden_states'][-1],
                                                                               device, event_transform)
                    lm_logits = output[0]
                    ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
                    loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), output_ids.view(-1))
                    loss = loss.view(output_ids.size(0), -1)

                    storylines = model.generate(input_ids, attention_mask=input_masks,
                                                max_length=args.gen_storyline_len, num_beams=args.num_beams)
                    if args.mu_switch:
                        p = - global_step / ((args.mu_epoch / args.num_train_epochs) * t_total) + 1
                    else:
                        p = args.mu / (args.mu + np.exp(global_step / args.mu)) if args.mu != 0 else 0.0
                    use_pred_mask = torch.ones(len(instance_indices), device=output_ids.device)
                    if use_story_input:
                        storylines = make_new_story_input(storylines[:, 2:], output_ids, tokenizer, all_story_inputs,
                                                          instance_indices, thresh=p)
                    else:
                        storylines = storylines[:, 1:]

                    storyline_masks = torch.where(storylines == 1, 0, 1)

                    decoder_output_ids_s = shift_tokens_right(output_ids_s, tokenizer.pad_token_id)
                    output = story_model(storylines, attention_mask=storyline_masks,
                                        decoder_input_ids=decoder_output_ids_s, output_hidden_states=True)

                    output_ids_s_event_index = convert_scalar_index_to_tuple(output_ids_s_index)
                    output_ids_s_event_representation = get_event_representation(output_ids_s_event_index,
                                                                                 output['decoder_hidden_states'][-1],
                                                                                 device, event_transform)

                    cross_ot_loss, cross_ot_allocate, _ = ipot_distance_loss(
                        output_ids_event_representation, output_ids_s_event_representation, device,
                        beta=args.beta, gamma=args.gamma, iteration=args.iteration)
                    cross_ot_loss = torch.mean(cross_ot_loss)
                    cross_ot_allocate_bak = cross_ot_allocate.clone().detach()

                    lm_logits_s = output[0]
                    loss_s = ce_loss_fct(lm_logits_s.view(-1, lm_logits_s.shape[-1]), output_ids_s.view(-1))
                    loss_s = loss_s.view(output_ids_s.size(0), -1)

                    reward = loss_s.clone().detach()
                    loss_bak = loss.clone().detach()
                    cnter_loss_s = cnt(loss_s)
                    loss_s = loss_s.sum() / sum(cnter_loss_s)
                    cnter_loss = cnt(loss)
                    loss_scaling_matrix = fine_grained_for_ot(args, loss_bak, output_ids, reward, output_ids_s_index,
                                                              cross_ot_allocate_bak, use_pred_mask)
                    loss_scaling_matrix.require_grad = False
                    compensate_coefficient = 1.0
                    loss = loss * loss_scaling_matrix * compensate_coefficient
                    loss = loss.sum() / sum(cnter_loss)

                    """
                    # change to this for comparative experiment in RL
                    reward = loss_s.clone()
                    loss_s = loss_s.mean()
                    reward = reward.view(output_ids.size(0, -1))
                    cnter = cnt(reward)
                    reward = reward.sum(dim=-1)
                    reward = reward/cnter
                    reward = reward/reward.sum()
                    reward.require_grad = False
                    loss = loss*reward.unsqueeze(-1)
                    loss = loss.mean(-1).sum()
                    """


                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                    loss_s = loss_s.mean()
                    cross_ot_loss = cross_ot_loss.mean()
                    # regularization_loss = regularization_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    loss_s = loss_s / args.gradient_accumulation_steps
                    cross_ot_loss = cross_ot_loss / args.gradient_accumulation_steps

                alpha = args.alpha
                loss += loss_s
                loss += alpha * cross_ot_loss

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += output_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if step > 0 and step % 1000 == 0:
                    logger.info("current train loss is %s" % (tr_loss / float(nb_tr_steps)))
                    logger.info("global_step: {}, mu: {}".format(global_step, p))

            if args.do_eval:

                eval_inputs = select_field(eval_features, 'inputs')
                eval_encoded_inputs = tokenizer(eval_inputs, padding=True, truncation=True, return_tensors="pt")

                eval_input_ids = eval_encoded_inputs['input_ids']
                eval_input_mask = eval_encoded_inputs['attention_mask']

                eval_labels = select_field(eval_features, 'labels')
                eval_encoded_outputs = tokenizer(eval_labels, padding=True, truncation=True, return_tensors="pt")
                eval_output_ids = eval_encoded_outputs['input_ids']
                eval_output_mask = eval_encoded_outputs['attention_mask']

                eval_stories = select_field(eval_features, 'story')
                eval_outputs_stories = tokenizer(eval_stories, padding=True, truncation=True, return_tensors="pt")

                eval_output_ids_stories = eval_outputs_stories['input_ids']
                eval_output_ids_stories[eval_output_ids_stories == tokenizer.pad_token_id] = -100
                eval_output_mask_stories = eval_outputs_stories['attention_mask']

                if use_story_input:
                    eval_story_inputs = select_field(eval_features, 'story_input')
                eval_key_indices = torch.tensor(list(range(len(eval_labels))), dtype=torch.long)

                eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_key_indices, eval_output_ids,
                                          eval_output_mask, eval_output_ids_stories, eval_output_mask_stories)

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                story_model.eval()

                perplexity = []
                eval_loss = []
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_masks, instance_indices, output_ids, output_masks, output_ids_s, output_masks_s = batch

                    with torch.no_grad():
                        storylines = model.generate(input_ids, attention_mask=input_masks,
                                                    max_length=args.gen_storyline_len, num_beams=args.num_beams)
                        if use_story_input:
                            storylines = make_new_story_input(storylines[:, 2:], output_ids, tokenizer,
                                                              eval_story_inputs, instance_indices)
                        else:
                            storylines = storylines[:, 1:]
                        storyline_masks = torch.where(storylines == 1, 0, 1)
                        eval_loss_s = story_model(storylines, attention_mask=storyline_masks,
                                                  labels=output_ids_s, decoder_attention_mask=output_masks_s)[0]
                        perplexity.append(torch.exp(eval_loss_s).item())

                logger.info("Sequence Perplexity %.4f" % np.mean(perplexity))

                if np.mean(perplexity) < best_eval_perplexity:
                    best_eval_perplexity = np.mean(perplexity)
                    logger.info("Save at Epoch %s" % epoch)
                    with open(output_perf_file, 'w') as outfile:
                        outfile.write("%.4f" % best_eval_perplexity)
                    if args.save_model:
                        torch.save(model_to_save.state_dict(), output_model_file)
                        torch.save(model_to_save_s.state_dict(), output_model_file_s)
                model.train()
                story_model.train()


if __name__ == "__main__":
    main()
