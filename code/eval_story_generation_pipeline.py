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
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, BartForConditionalGeneration
from utils import *
from optimization import *
from pathlib import Path
import json
from nlgeval import NLGEval

# 同样是设置logging的打印格式，默认采用的是打印在控制台上的streamhandler,包括了输出控制台的打印格式
# 其中包括了打印显示的时间、处理的层级、处理模块的名称、最终要打印的信息等等。
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
# 实例化模块的名称
logger = logging.getLogger(__name__)
PYTORCH_PRETRAINED_ROBERTA_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_ROBERTA_CACHE',
                                                  Path.home() / '.pytorch_pretrained_roberta'))

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    # 用于测试的数据的路径
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    # 采用的预训练模型
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="pre-trained model selected in the list: roberta-base, "
                             "roberta-large, bert-base, bert-large. ")
    # 当前执行的任务的名称
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    # 存储的路径的位置
    parser.add_argument("--file_suffix",
                        default=None,
                        type=str,
                        required=True,
                        help="unique identifier for data file")
    ## Other parameters
    # 最大的故事长度
    parser.add_argument("--max_seq_length",
                        default=320, # 8 * 8 * 5
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    # 最大的故事线的长度
    parser.add_argument("--gen_storyline_len",
                        default=128,  # 8 * 8 * 5
                        type=int,
                        help="The maximum length for generated storyline.")
    # 模型保存的位置
    parser.add_argument("--model_dir",
                        type=str,
                        help="saved model dir",
                        default="")
    # 采样前K个
    parser.add_argument("--topk_sample",
                        action='store_true')
    # 模型是否只接受小写的输入
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    # 用于验证的batchsize的大小
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    # 中间验证的隐藏层大小
    parser.add_argument("--mlp_hid_size",
                        default=64,
                        type=int,
                        help="hid dimension for MLP layer.")
    # 学习率的设定
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    # 训练了多少个epoch
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    # 线性学习率调整的epoch比例
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    # 是否使用cuda的设备
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    # 是否采用structured storyline model
    parser.add_argument("--no_struct",
                        action='store_true',
                        help="Whether not to use structure in the input")
    # 多GPU的情况下如何做分卡的调整
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    # 当前设定的随机种子
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    # 梯度累积步数
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    # 是否采用混合精度的方式进行训练
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    # 搭配混合精度训练需要使用的
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    # 采用第几个cuda
    parser.add_argument('--cuda',
                        type=str,
                        default="",
                        help="cuda index")
    # 采用第几个设备
    parser.add_argument('--device_num',
                        type=str,
                        default="0",
                        help="cuda device number")
    # 构成的故事线要涉及几个故事
    parser.add_argument('--input_event_num',
                        type=int,
                        default=1,
                        help="input event number")
    # 这里不太懂是什么意思
    parser.add_argument('--no_label',
                        action='store_true',
                        help="predict unlabeled data")
    args = parser.parse_args()

    # 按照数字的标号去分配GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # 设定当前使用第几块GPU
    torch.cuda.set_device(torch.device('cuda:%s' % args.device_num))

    # 基本的一些设定(是否使用结构化的输入，是否用story_input构成额外故事线等等)
    use_relation = True if "with_rel" in args.model_dir else False
    no_struct = True if "no_struct" in args.model_dir else False
    use_story_input = True if "story_input" in args.model_dir else False
    template = True if "event_script" in args.model_dir else False

    # 也是判定是否使用cuda的，这里不用管
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

    # 当前的实验环境设定
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    # 是否进行梯度累积的设定
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    # 所有随机种子的设定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # 打印当前正在执行的任务
    task_name = args.task_name.lower()
    logger.info("current task is " + str(task_name))

    # 当前所使用模型的目录加载地址
    logger.info(args.model_dir)
    # 加载训练好的预训练模型
    model_state_dict = torch.load(args.model_dir + "pytorch_model.bin", map_location=device)
    model_state_dict_s = torch.load(args.model_dir + "pytorch_model_s.bin", map_location=device)


    # 加载和训练时候用的相同的分词器，这里的分词器用的就是model_state_dict有点奇怪,感觉要state_dict是没什么用的
    tokenizer = AutoTokenizer.from_pretrained(args.model, state_dict=model_state_dict)
    if no_struct:
        num_added_toks = tokenizer.add_tokens(['<eoe>'])
    else:
        # 同样加入一些token,是给分词器加入这些token，表明遇到这些token要进行相应的分词
        num_added_toks = tokenizer.add_tokens(['<eoe>', '<before>', '<after>', '<vague>'])

    logger.info('We have added %s tokens' % num_added_toks)

    # 载入两个模型
    model = BartForConditionalGeneration.from_pretrained(args.model)
    story_model = BartForConditionalGeneration.from_pretrained(args.model)

    # 为新加入的两个token给出embedding
    model.resize_token_embeddings(len(tokenizer))
    story_model.resize_token_embeddings(len(tokenizer))

    # 加载模型的参数
    model.load_state_dict(model_state_dict)
    story_model.load_state_dict(model_state_dict_s)

    # 模型搬运到对应的GPU设备上
    model.to(device)
    story_model.to(device)
    # 释放内存空间
    del model_state_dict
    del model_state_dict_s

    # Prepare optimizer
    param_optimizer = list(model.named_parameters()) + list(story_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # decay的对象
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

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
        # 采用的优化器
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=1)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    for split in ['test']:
        # 打开相应的测试集模块
        with open("%s%s%s" % (args.data_dir, split, args.file_suffix)) as infile:
            eval_data = json.load(infile)

        # 读取测试集的相应特征
        if 'wp' in args.file_suffix and not template:
            eval_features = convert_to_wp_features(eval_data, input_event_num=args.input_event_num,
                                                    use_relation=use_relation, no_struct=no_struct, is_eval=True,
                                                    use_story_input=use_story_input)
        elif 'wp' in args.file_suffix and template:
            eval_features = convert_to_event_script_wp_features(eval_data, input_event_num=0, use_relation=use_relation,
                                                                no_struct=no_struct, use_story_input=True, is_eval=True)
        elif 'wp' not in args.file_suffix and template:
            eval_features = convert_event_script_to_input(eval_data, input_event_num=0, use_relation=use_relation,
                                                          no_struct=no_struct, use_story_input=True, is_eval=True)
        else:
            eval_features = convert_graph_to_features(eval_data, input_event_num=0, use_relation=use_relation,
                                                      no_struct=no_struct, use_story_input=True, is_eval=True)

        # 进行分词，是list(list)的形式，每一个list都是这些storyline对应的token id
        eval_inputs = select_field(eval_features, 'inputs')
        eval_encoded_inputs = tokenizer(eval_inputs, padding=True, truncation=True, return_tensors="pt")

        # 进行分词，得到一系列的list(list)对象，得到的是测试集的待输入的<MASK>之后的故事线,'pt'本质上表明返回的数据类型是tensor类型
        eval_input_ids = eval_encoded_inputs['input_ids']
        eval_input_mask = eval_encoded_inputs['attention_mask']

        # 进行分词，是list(list)的形式，每一个list都是这些storyline对应的token id
        eval_labels = select_field(eval_features, 'labels')
        eval_encoded_outputs = tokenizer(eval_labels, padding=True, truncation=True, return_tensors="pt")

        # 得到的是测试集的storyline的groundtruth
        eval_output_ids = eval_encoded_outputs['input_ids']
        eval_output_mask = eval_encoded_outputs['attention_mask']

        # 进行分词，是list(list)的形式，每一个list都是这些storyline对应的token id
        eval_stories = select_field(eval_features, 'story')
        # 修正了这里，确保它的长度是不会溢出的
        eval_outputs_stories = tokenizer(eval_stories, padding=True, truncation=True, return_tensors="pt",
                                         max_length=1024)

        # 得到的是测试集的story的groundtruth
        eval_output_ids_stories = eval_outputs_stories['input_ids']

        eval_output_ids_stories[eval_output_ids_stories == tokenizer.pad_token_id] = -100
        # 获得了对应的attention
        eval_output_mask_stories = eval_outputs_stories['attention_mask']

        # 取出对应的story input
        eval_story_inputs = select_field(eval_features, 'story_input')

        # 同样是对其进行索引标记，确保每一个batch可知它们来自于哪一些样本
        eval_key_indices = torch.tensor(list(range(len(eval_labels))), dtype=torch.long)

        # 打印处理好的数据集信息
        logger.info("id_size: {}, mask_size: {}, instance_key_size: {}, label_size: {}".format(
            eval_input_ids.size(), eval_input_mask.size(), eval_key_indices.size(), eval_output_ids.size()))

        # 封装成TensorDataset数据集
        data = TensorDataset(eval_input_ids, eval_input_mask, eval_key_indices, eval_output_ids,
                             eval_output_mask, eval_output_ids_stories, eval_output_mask_stories)

        # 直接进行预测，用SequentialSampler即可
        eval_sampler = SequentialSampler(data)

        # 载入对应的DataLoader
        eval_dataloader = DataLoader(data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # preds是一个list(list)数据结构，内层的list包含的是一个batch中所有预测的故事，且每一个故事都是一个字符串存储在list中
        # golds是一个list(list)数据结构，内层的list包含的是一个batch中所有groundtruth的故事，且每一个groundtruth都是一个字符串存储在list中
        # perplexity记录的每一个batch中的perplexity的大小
        preds, golds, events, perplexity = [], [], [], []
        # pred_to_eval是一个list数据结构，list中的每一项都是一个样本真实从prediction到evaluation要进行比对的地方
        # gold_to_eval是一个list数据结构,list中的每一项都是一个样本真实从evaluation到prediction要进行比对的地方
        pred_to_eval, gold_to_eval = [], []
        model.eval()
        # contexts是一个list(list)数据结构，内层的list包含的是一个batch中所有groundtruth的storyinput，每一个groundtruth storyinput都是一个字符串存在list中
        contexts = []

        wrong_structure, wrong_gold = 0, 0

        perplexity, gen_perplexity, overlaps, overlap_ts, overlap_as = [], [], [], [], []
        repeat, distinct, repeat_f, distinct_f = [[] for _ in range(4)], [[] for _ in range(4)], \
                                                 [[] for _ in range(4)], [[] for _ in range(4)]

        gen_all_perp, gen_avg_perp, gen_perplexity, token_len = [], [], [], []
        pred_storylines = []
        missing_counter = 0
        # 取出相应的batch数据集
        storyline_temporal_count = dict()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # 转移到对应的GPU设备上
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, instance_indices, output_ids, output_masks, output_ids_s, output_masks_s = batch

            # 算baseline的时候需要用上这个
            if 'wp' not in args.file_suffix:
                # 构建output_ids_s，nullify的意思是使得什么作废，这里的具体意思是mask掉最终生成story的groundtruth的第一个句子，因为在故事线的story_input中其实已经用到了这些句子
                output_ids_s = nullify_story_inputs(output_ids_s, tokenizer, eval_story_inputs, instance_indices)

            with torch.no_grad():

                # 模型直接生成相应的故事线
                storylines = model.generate(input_ids, attention_mask=input_masks, max_length=args.gen_storyline_len)
                # 记录当前batch预测出的故事线，batch_decode的结果是用一个list进行存储的，list中的每一项存储的都是一个字符串
                pred_storylines.extend(tokenizer.batch_decode(storylines, skip_special_tokens=True))

                if use_story_input:
                    # 若使用story_input则构造新的故事线作为输入
                    storylines = make_new_story_input(storylines[:, 2:], output_ids, tokenizer,
                                                      eval_story_inputs, instance_indices)
                else:
                    storylines = storylines[:, 1:]

                # 同时要mask掉新的故事线中存在的padding，因为在拼接真实的，完整的故事线过程中不存在对batch的tokenizer的操作，因此不会自动做padding的补全
                storyline_masks = torch.where(storylines == 1, 0, 1)

                # 这里算的是Mask Language Model的loss，那些被标记成-100的loss是不被计算的
                eval_loss_s = story_model(storylines, attention_mask=storyline_masks, labels=output_ids_s, decoder_attention_mask=output_masks_s)[0]

                # 用上述的loss得到对应的perplexity
                perplexity.append(torch.exp(eval_loss_s).item())

                # 是否要进行topk的采样
                if args.topk_sample:
                    res = story_model.generate(storylines, attention_mask=storyline_masks, max_length=1024,
                                               return_dict_in_generate=True, output_scores=True,
                                               do_sample=True)
                else:
                    # 生成最终的故事，实际在做生成的时候，是没有decoder_input_ids的，在训练的时候才会有这个输入,以dict的形式返回各项值
                    res = story_model.generate(storylines, attention_mask=storyline_masks, max_length=1024,
                                               return_dict_in_generate=True, output_scores=True)
                # 生成的故事真实的故事
                # 是一个list，list的每一项是一个字符串组成的故事
                batch_preds = tokenizer.batch_decode(res.sequences, skip_special_tokens=True)
                # 实际输出故事的groundtruth
                batch_golds = [eval_stories[x] for x in instance_indices.tolist()]
                # 实际输入故事线的groundtruth
                batch_context = [eval_inputs[x] for x in instance_indices.tolist()]

                for i, (pred, gold, context) in enumerate(zip(batch_preds, batch_golds, batch_context)):
                    # 每一个生成的故事的token的长度
                    token_len.append(len(pred.split(' ')))
                    # 对预测的句子做句子的切分，返回的是一个list
                    sents = sep_output_sents(pred)
                    # 对真实的groundtruth的句子进行句子的切分，返回的是一个list
                    gold = sep_output_sents(gold)

                    if len(sents) == 1 or len(gold) == 1:
                        missing_counter += 1
                        continue

                    if 'wp' in args.file_suffix:
                        # 这里不考虑第一个句子，因为在storyline制备的时候已经加上去了
                        pred_to_eval.append(pred)
                        start = 0
                    else:
                        # 实际的从eval到groundtruth要考量的句子
                        pred_to_eval.append(' '.join(sents[1:]))
                        start = 1
                    # 实际从groundtruth到eval要考量的句子
                    gold_to_eval.append(' '.join(gold[start:]))

                    # 遍历1~4就可以得到当前句子的所有1grams,2grams,3grams,4grams
                    for j in range(4):
                        ngrams = collect_all_ngrams_flatten(sents[start:], n=j+1)
                        if len(ngrams) == 0:
                            # missing_counter += 1
                            continue
                        # 记录当前对应sentences的ngrams是否有重复项
                        repeat_f[j].append(contains_repeat(ngrams))
                        # 记录当前对应sentences的ngrams的重复率有多高
                        distinct_f[j].append(cal_distinct(ngrams))
                        # batch_preds是一个list，list中的每一项都是这个batch中用字符串组成的故事
                preds.extend(batch_preds)
                # batch_golds是一个list，list中的每一项都是这个batch中用字符串组成的groundtruth故事
                golds.extend(batch_golds)
                # batch_context也是一个list，list中的每一项都是这个batch中用字符串组成的groundtruth的经过MASK的storyline输入
                contexts.extend(batch_context)

                # 本质上都是batch中的同步处理，长度理应相同
        assert len(preds) == len(golds)
        assert len(pred_storylines) == len(golds)

        # 文件名称，把最后的/去掉了
        filename = args.model_dir.split("event_0")[-1][:-1]

        out_dir = 'generation_wp' if 'wp' in args.file_suffix else 'generation'

        if args.topk_sample:
            filename += "_topk_sample"
        with open("../%s/storylines_from%s_%s.json" % (out_dir, filename, args.seed), 'w') as outfile:
            # 存进预测的故事线
            json.dump(pred_storylines, outfile)

        with open("../%s/stories_from%s_%s.json" % (out_dir, filename, args.seed), 'w') as outfile:
            # 存进写好的故事
            json.dump(preds, outfile)

        logger.info("Total %d wrong structures" % wrong_structure)

        # assert len(repeat_f[0]) == len(distinct_f[0]) == len(golds) - missing_counter

        # 生成的故事线的平均perplexity
        logger.info("Gold Sequence Perplexity %.4f" % np.mean(perplexity))
        # 平均的token_len
        logger.info("Average Token Length %.4f" % np.mean(token_len))
        logger.info("==========")
        # 从1gram到4grams的信息
        for k in range(4):
            logger.info("Flatten Repeat-%s %.4f" % (k+1, np.mean(repeat_f[k])))
            logger.info("Flatten Distinct-%s %.4f" % (k+1, np.mean(distinct_f[k])))
            logger.info("==========")
        logger.info("Missing Ratio %.4f" % (missing_counter / float(len(golds))))

        # evaluating
        print('Start evaluating ...')
        nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)
        # 显示最终的评测指标计算结果
        metrics_dict = nlgeval.compute_metrics([gold_to_eval], pred_to_eval)
        print(metrics_dict)


if __name__ == "__main__":
    main()
