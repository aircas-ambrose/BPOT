"""
This file is computing all metrics for storyline and story
"""
from tqdm import tqdm
import string
from collections import Counter
import json
import spacy
import os
import torch
import nltk
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from nlgeval import NLGEval
import argparse
import nltk

nlp = spacy.load('en_core_web_sm')
nltk_ana = nltk.data.load('tokenizers/punkt/english.pickle')

def cal_distinct(ngrams):
    """
    :param ngrams:
    :return:
    """
    return len(set(ngrams)) / float(len(ngrams))


def contains_repeat(ngrams):
    """
    :param ngrams:
    :return:
    """
    return False if len(set(ngrams)) == len(ngrams) else True


def sep_output_sents(output):
    return nltk.tokenize.sent_tokenize(output)

def collect_all_ngrams_flatten(sents, n=4):
    ngrams = []
    sent = ' '.join(sents)
    tokens = sent.split(' ')
    if len(tokens) >=4 :
        for i in range(len(tokens)-n+1):
            ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams

def collect_all_ngrams(sents, n=4):
    ngrams = []
    for sent in sents:
        tokens = sent.split(' ')
        if len(tokens) >= n:
            for i in range(len(tokens)-n+1):
                ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams


def compute_global_distinct_ratio_story(file_path):
    """计算故事全局的distinct ratio"""
    story_data = json.load(open(file_path, 'r'))
    tokens_counter = Counter()
    print('>'*89)
    print('computing the global distinct ratio for {} story in {} file'.format(len(story_data), file_path))
    for story in tqdm(story_data):
        story_token = story.split()
        tokens_counter.update(story_token)
    # print('Total Vocab: {}, Vocab: Token Ratio: {}'.format(len(tokens_counter),
    #                                                        100 * (distinct_count / sum(tokens_counter.values()))))
    distinct_count = 0
    for key, value in tokens_counter.items():
        if value == 1:
            distinct_count += 1
    print('Total Vocab: {}, Vocab: Token Ratio: {}'.format(len(tokens_counter), (distinct_count /
                                                           sum(tokens_counter.values())) * 100))

def compute_local_distinct_ratio_story(file_path):
    story_data = json.load(open(file_path, 'r'))
    tgrams, fgrams = Counter(), Counter()
    print('computing the local distinct ratio for {} story in {} file'.format(len(story_data), file_path))
    for sample_story_data in tqdm(story_data):
        sents = nltk.tokenize.sent_tokenize(sample_story_data)
        sents = [sample_sents.strip() for sample_sents in sents]
        tgrams.update(collect_all_ngrams(sents, 3))
        fgrams.update(collect_all_ngrams(sents, 4))
    print('>' * 89)
    print('Total 3 Grams: {}, Distinct 3 Grams Ratio: {}'.format(len(tgrams), (len(tgrams) / sum(tgrams.values())) * 100))
    print('>' * 89)
    print('Total 4 Grams: {}, Distinct 4 Grams Ratio: {}'.format(len(fgrams), (len(fgrams) / sum(fgrams.values())) * 100))

def compute_mt_metric(pred_path, gold_path, wp=False):
    pred = json.load(open(pred_path, 'r'))
    gold_dataset = json.load(open(gold_path, 'r'))
    gold = [example['story'].lower() for example in gold_dataset]
    token_len, pred_to_eval, gold_to_eval = [], [], []
    missing_counter = 0
    print('computing the mt metric for {} stories in {} file'.format(len(pred), pred_path))
    for i, (sample_pred, sample_gold) in enumerate(zip(pred, gold)):
        token_len.append(len(sample_pred.split(' ')))
        sents = sep_output_sents(sample_pred)
        golds = sep_output_sents(sample_gold)

        if len(sents) == 1 or len(golds) == 1:
            missing_counter += 1
            continue

        if wp:
            pred_to_eval.append(sample_pred)
            start = 0
        else:
            pred_to_eval.append(' '.join(sents[1:]))
            start = 1
        gold_to_eval.append(' '.join(golds[start:]))

    print('start evaluating ...')
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)
    metrics_dict = nlgeval.compute_metrics([gold_to_eval], pred_to_eval)
    print('>' * 89)
    print(metrics_dict)

def compute_repeat(pred_path, wp=False):
    pred = json.load(open(pred_path, 'r'))
    repeat_f = [[] for _ in range(4)]
    whole_token_len = []
    for i, sample_pred in enumerate(pred):
        whole_token_len.append(len(sample_pred.split(' ')))
        sents = sep_output_sents(sample_pred)
        if wp:
            start = 0
        else:
            start = 1
        for j in range(4):
            ngrams = collect_all_ngrams_flatten(sents[start:], n=j+1)
            repeat_f[j].append(contains_repeat(ngrams))

    print('computing the repeat metric for {} story in {} file'.format(len(pred), pred_path))
    print('>'*89)
    print('===========')
    for k in range(4):
        print('Flatten Repeat-%s %.4f' % (k+1, np.mean(repeat_f[k])))
        print('==========')
    print('The token len is %.4f' % np.mean(whole_token_len))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default="gpt2",
                        type=str)
    parser.add_argument("--device_num",
                        default="4",
                        type=str)
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int)
    parser.add_argument("--seed",
                        default=5,
                        type=int)
    args = parser.parse_args()
    base_path = ''   #  the directory of generated results
    gold_path = ''   #  the path for golden story

    story_file_path = ['']
    storyline_file_path = ['']

    for story_file_path, storyline_file_path in zip(story_file_path, storyline_file_path):
        compute_local_distinct_ratio_story(os.path.join(base_path, story_file_path))
        compute_mt_metric(os.path.join(base_path, story_file_path), gold_path, True)
        # set wp for whether testing WritingPrompt datasets
        compute_repeat(os.path.join(base_path, story_file_path), wp=False)



