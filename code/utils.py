from collections import defaultdict, Counter, OrderedDict
from typing import List, Mapping, Union
from datetime import datetime
import logging
import torch
import pandas as pd
#from scipy.stats import mode
from collections import Counter
import random
import json
import numpy as np
import re
import nltk
import torch.nn as nn
import torch.nn.functional as F
# nltk.download('punkt')
# from nltk.translate.bleu_score import SmoothingFunction
# from nltk.translate.bleu_score import sentence_bleu

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

label_map = OrderedDict([('VAGUE', 'VAGUE'),
                         ('BEFORE', 'BEFORE'),
                         ('AFTER', 'AFTER'),
                         ('SIMULTANEOUS', 'SIMULTANEOUS')
                        ])

rel_to_index = {'before': 0, 'after': 1, 'vague': 2, '<unk>': 3}
index_to_rel = {0: 'before', 1: 'after', 2: 'vague', 3: '<unk>'}

sep_token, eoe_token = ' ; ', '<eoe>'

class EventTransform(nn.Module):

    def __init__(self, hidden_dim):
        super(EventTransform, self).__init__()
        self.event_transformer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, event_representation):
        return self.event_transformer(event_representation)

# 定义用于抽取事件级别表达的模型
class EventExtractionModel(nn.Module):

    def __init__(self, hidden_dim):
        super(EventExtractionModel, self).__init__()
        self.trigger_transformation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.subject_transformation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.object_transformation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.event_transformation = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.activation = nn.Tanh()

    def forward(self, trigger_representation, subject_representation, object_representation):
        event_representation = self.activation(self.event_transformation(
            self.trigger_transformation(trigger_representation) +
            self.subject_transformation(subject_representation) +
            self.object_transformation(object_representation)
        ))
        # 返回得到的各个事件级别的表达
        return event_representation


# 做一个在事件层级进行交互的模型
class EventInteractionModel(nn.Module):

    def __init__(self, hidden_dim):
        super(EventInteractionModel, self).__init__()
        self.interaction = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1,
                                  bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.activation = nn.Tanh()
        self.transformation = nn.Linear(2*hidden_dim, 2*hidden_dim, bias=True)
        self.gate = nn.GLU()

    def forward(self, x):
        x = self.interaction(x)
        x = self.activation(self.transformation)
        x = self.gate(x)
        return x


# 计算生成的故事中实际story的覆盖率
def cal_event_story_overlaps(context, sents):
    # events是生成的故事中某一个故事的storyline
    events = context.split(eoe_token)[:-1]
    assert len(events) == len(sents) == 5
    overlap, overlap_t, overlap_a = 0, 0, 0
    for i, event in enumerate(events):
        # 取出一个storyline中的每一个event
        event = event.split(sep_token)
        if event[0] in sents[i]:
            # trigger的重合度
            overlap_t += 1
            # 两个argument的重合度
        if event[1] in sents[i] and event[-1] in sents[i]:
            overlap_a += 1
            # 整体重合的程度
        if all([e in sents[i] for e in event]):
            overlap += 1
    return float(overlap) / len(events), float(overlap_t) / len(events), float(overlap_a) / len(events)


def collect_all_ngrams(sents, n=4):
    ngrams = []
    for sent in sents:
        tokens = sent.split(' ')
        if len(tokens) >= n:
            for i in range(len(tokens)-n+1):
                ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams


def collect_all_ngrams_flatten(sents, n=4):
    """
    :param sents: 输入的预测生成的句子
    :param n: 设定要获取当前句子的多少gram
    :return:  返回当前句子中包含的所有的ngrams
    """
    # 最终要返回的是一个列表
    ngrams = []
    # 拼接句子
    sent = ' '.join(sents)
    # 按照空格切分相应的token
    tokens = sent.split(' ')
    # 只有token的数量在大于4的时候才会进行计算，依次序的切分对应的ngrams
    if len(tokens) >= 4:
        for i in range(len(tokens)-n+1):
            ngrams.append(' '.join(tokens[i:i+n]))   # 切分4grams
    return ngrams


def contains_repeat(ngrams):
    """
    :param ngrams: 输入的是一个列表，里面包含当前句子中所有的ngrams
    :return: 返回标记，判断是否有重复的ngrams
    """
    # 无重复标记则返回False，否则返回True
    return False if len(set(ngrams)) == len(ngrams) else True


def cal_distinct(ngrams):
    """
    :param ngrams: 输入的是一个列表
    :return: 返回当前sentence的ngrams的重复程度
    """
    return len(set(ngrams)) / float(len(ngrams))


# 触发词抽取
def extract_triggers(storyline):
    triggers = [x.split(" ; ")[0] for x in storyline.split(eoe_token)]
    return triggers[:5] if len(triggers) > 5 else triggers


# 计算其中的多样性得分
def compute_diversity_scores(storylines):
    scores = [len(set(extract_triggers(storyline))) / 5.0 for storyline in storylines]
    return np.mean(scores)


def populate_loss(losses, tr_loss):
    total_loss = 0.0
    for i, loss in enumerate(losses):
        total_loss += loss
        tr_loss[i] += loss.item()
    return total_loss, tr_loss


def populate_losses(loss_t, loss_a1, loss_a2):
    total_loss_t, total_loss1, total_loss2 = 0.0, 0.0, 0.0
    for lt, l1, l2 in zip(loss_t, loss_a1, loss_a2):
        total_loss_t += lt
        total_loss1 += l1
        total_loss2 += l2
    return total_loss_t, total_loss1, total_loss2


def decode_batch_outputs(all_event_arguments, tokenizer, batch_size):
    outputs = ["" for _ in range(batch_size)]
    for t in range(5):
        cur_text = tokenizer.batch_decode(all_event_arguments[t], skip_special_tokens=True)
        for b in range(batch_size):
            outputs[b] += cur_text[b] + ' <eoe> '
    return outputs


def decode_batch_outputs_sep(all_event_arguments, tokenizer, batch_size):
    outputs = ["" for _ in range(batch_size)]
    for t in range(5):
        cur_text = tokenizer.batch_decode(all_event_arguments[t], skip_special_tokens=True)
        for b in range(batch_size):
            outputs[b] += ' ; '.join(cur_text[b * 3:(b + 1) * 3]) + ' <eoe> '
    return outputs

def cal_f1(pred, gold):
    cor, total_pred = 0.0, 0.0
    for p, g in zip(pred, gold):
        if p == g:
            cor += 1
        if p != 3:
            total_pred += 1
    recl = cor / len(gold)
    prec = cor / total_pred if total_pred > 0.0 else 0.0
    return 0.0 if recl + prec == 0.0 else 2.0 * recl * prec / (recl + prec)


def create_selection_index(input_ids, sep_ids, num_input_events):
    selection_index = [[] for _ in range(num_input_events)]
    batch_size = input_ids.size()[0]
    for b in range(batch_size):
        sep_locs = (input_ids[b] >= sep_ids[0]).nonzero(as_tuple=True)[0]
        for ie in range(num_input_events):
            event_start = 3*ie
            for ia in [0, 1, 2]:
                arg_start = sep_locs[event_start+ia].item() + 1
                arg_end = sep_locs[event_start+ia+1].item() - 1
                selection_index[ie].append([b, arg_start, arg_end])
    return selection_index


def create_tok_selection_index(input_ids, num_input_events, starts, ends):
    selection_index = [[] for _ in range(num_input_events)]
    batch_size = input_ids.size()[0]
    for b in range(batch_size):
        for ie in range(num_input_events):
            for ia in [0, 1, 2]:
                arg_start = starts[b, 3*ie+ia]
                arg_end = ends[b, 3*ie+ia]
                selection_index[ie].append([b, arg_start, arg_end])
    return selection_index


def convert_sequences_to_features(data, input_event_num=1):
    # input: Event 1
    # output: R(1,2) ; Event 2 ; R(2,3) ; Event 3 ; R(3,4) ; Event 4 ; R(4,5) ; Event5

    def make_event_input(event):
        return "%s %s %s %s %s %s" % (event[0], sep_token, event[1], sep_token, event[2], eoe_token)

    samples = []
    counter = 0
    for v in data:

        input = [v['title'], eoe_token]
        for e in v['events'][:input_event_num]:
            input += [make_event_input(e)]

        output = []
        for e in v['events'][input_event_num:]:
            output += [make_event_input(e)]

        sample = {'inputs': ' '.join(input),
                  'labels': ' '.join(output),
                  'relations': [rel_to_index[r.lower()] for r in v['relations']]}

        counter += 1
        # check some example data
        if counter < 5:
            print(sample)

        samples.append(sample)

    return samples

def convert_graph_to_features_gpt(data, input_event_num=0, use_relation=False, is_eval=False,
                              no_struct=False, use_story_input=False):
    # 依然是只保留第一个事件
    def get_unmask_events(N, num):
        return [[0] + random.Random(n).sample(range(1, 5), num) for n in range(N)]

    samples = []
    counter = 0

    unmask_events = get_unmask_events(len(data), input_event_num)
    assert len(unmask_events) == len(data)

    for v, umask in zip(data, unmask_events):
        # 开始组织相应的mask操作
        sample = {}
        # 对应的样本
        if is_eval:
            keys = ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
            if keys[0] in list(v.keys()):
                story = []
                for key in keys:
                    story.append(v[key].lower())
                sample['story'] = ' '.join(story)
            else:
                sample['story'] = v['story'].lower()

        # 原来这里的story_input没有小写,但是前面已经转过小写了，所以这一步是没有关系的，把它的第一句话放上去
        if use_story_input:
            sample['story_input'] = nltk.tokenize.sent_tokenize(sample['story'])[0].lower()

        # check some example data
        if counter < 1:
            # 打印一个样本的样例
            logger.info(sample)

        counter += 1
        samples.append(sample)

    return samples


def convert_graph_to_features(data, input_event_num=0, use_relation=False, is_eval=False,
                              no_struct=False, use_story_input=False):
    """
    :param data: 输入的list(dict)格式的数据集
    :param input_event_num: 每一个story预先输出多少个event构成storyline
    :param use_relation: 是否使用temporal prompt
    :param is_eval: 是否进行eval
    :param no_struct: 是否使用structured storyline model
    :param use_story_input: 是否改变采用story_input
    :return:
    """   # sep token指的是用于分隔的token，注意这里的sep token前后是加了两个空格的,这里的sep token额外加了两个空格
    sep_token, eoe_token, mask_token = ' ; ', '<eoe>', '<mask>'

    def make_event_input(event, mask=False):
        '''把每一个example中的story做成mask标签的格式'''
        if not mask:
            trigger, a1, a2 = event[0].replace(';', ','), event[1].replace(';', ','), event[2].replace(';', ',')
            return "%s%s%s%s%s" % (trigger, sep_token, a1, sep_token, a2)
        else:
            return "%s%s%s%s%s" % (mask_token, sep_token, mask_token, sep_token, mask_token)

    def get_unmask_events(N, num):
        return [[0] + random.Random(n).sample(range(1, 5), num) for n in range(N)]

    samples = []
    counter = 0

    unmask_events = get_unmask_events(len(data), input_event_num)
    assert len(unmask_events) == len(data)

    for v, umask in zip(data, unmask_events):
        #input = [v['title'], eoe_token]
        input = []
        for i, (e, r) in enumerate(zip(v['events'], v['relations'] + ['eoe'])):
            if i > 0 and no_struct:
                break
            mask = False if i in umask else True
            r_mask = '<%s>' % r
            if use_relation:
                input += [make_event_input(e, mask), r_mask]
            else:
                input += [make_event_input(e, mask), eoe_token]

        output = []
        for e, r in zip(v['events'], v['relations'] + ['eoe']):
            if not no_struct:
                output += [make_event_input(e), '<%s>' % r]
            else:
                output += [make_event_input(e), eoe_token]

        sample = {'inputs': ''.join(input).lower(),
                  'labels': ''.join(output).lower()}

        if is_eval:
            sample['story'] = v['story'].lower()

        # 原来这里的story_input没有小写,但是前面已经转过小写了，所以这一步是没有关系的。
        if use_story_input:
            sample['story_input'] = nltk.tokenize.sent_tokenize(sample['story'])[0]

        # check some example data
        if counter < 1:
            # 打印一个样本的样例
            logger.info(sample)

        counter += 1
        samples.append(sample)

    return samples


def encode_all_stories(all_stories, tokenizer):
    """
    :param all_stories: 是一个list(list)数据结构，外层的list中的每一个元素都该表一个example,内层dict的每一个元素都代表该example的一个句子
    :param tokenizer: 用于分词的一个分词工具
    :return: 返回该example句子整体分好词的结果，以及分词的过程中对应每个句子分界处的索引位置
    """
    # 特别要值得注意的是，这里只求了索引对应的开头位置，没有求索引对应的结束位置
    all_stories_index_results = []
    story_index_max_len = 0
    for example_split_story in all_stories:
        split_sentence_token = tokenizer(example_split_story)
        example_story_index = []
        # 这里只需要在开头减一个1就可以，记录第一个sentence对应的索引位置，只求了对应分句的结束位置
        example_story_index.append(len(split_sentence_token['input_ids'][0][:-1]) - 1)
        for i in range(1, len(split_sentence_token['input_ids'])):
            example_story_index.append(example_story_index[i-1] + len(split_sentence_token['input_ids'][i][1:-1]))
        if len(example_story_index) > story_index_max_len:
            story_index_max_len = len(example_story_index)
        all_stories_index_results.append(example_story_index)
    for i in range(len(all_stories_index_results)):
        # 进行补全，保证它们的长度位置是一致的。
        all_stories_index_results[i].extend([-1 for _ in range(story_index_max_len - len(all_stories_index_results[i]))])
    all_stories_index_results = torch.tensor(all_stories_index_results, dtype=torch.int64)
    # 返回的是tokenize之后各个分句结束的位置，对应的是句号位置的索引
    return all_stories_index_results


def encode_all_stories_update(all_split_story, tokenizer):
    """
    :param all_stories: 是一个list(list)数据结构，外层的list中的每一个元素都该表一个example,内层dict的每一个元素都代表该example的一个句子
    :param tokenizer: 用于分词的一个分词工具
    :return: 返回分词封装好的结果以及对应的tokenize的结果
    """
    # input_ids
    output_ids_s = []
    # attention_mask
    output_attention_mask_s = []
    # 存放每一个story的切分结果
    stories_index = []
    # 取出每一个样本
    max_len = 0
    for sample_split_story in all_split_story:
        sample_token_ids_s = []
        sample_attention_mask_s = []
        sample_story_index = []
        # 取出样本中的每一个句子
        for index, sentence in enumerate(sample_split_story):
            # 算上除了第一个句子之外的空格,保证拆开的分词结果和全局的分词结果是保持一致的
            if index != 0:
                sentence = ' ' + sentence
            else:
                sentence = sentence
            # 对每一个句子进行分词
            sentence_token = tokenizer.tokenize(sentence)
            # 分词的结果存放
            sample_token_ids_s.extend(sentence_token)
            # 索引本身从0开始，又要加上最开始的开始符号<s>，固不用+1
            sample_story_index.append(len(sample_token_ids_s))
            for j in sentence_token:
                # 加入相应的attention mask
                sample_attention_mask_s.append(1)
        # 转化成token
        sample_output_id_s = tokenizer.convert_tokens_to_ids(sample_token_ids_s)
        # 插入起始符
        sample_output_id_s.insert(0, 0)
        # 插入起始符号的attention mask
        sample_attention_mask_s.insert(0, 1)
        # 加入结束符号</s>
        sample_output_id_s.insert(len(sample_output_id_s), 2)
        # 加入结束符号的attention mask
        sample_attention_mask_s.insert(len(sample_attention_mask_s), 1)
        assert len(sample_attention_mask_s) == len(sample_output_id_s)
        if len(sample_attention_mask_s) > max_len:
            max_len = len(sample_attention_mask_s)
        output_ids_s.append(sample_output_id_s)
        output_attention_mask_s.append(sample_attention_mask_s)
        stories_index.append(sample_story_index)
    # 补全相应的最大长度
    for i in range(len(output_ids_s)):
        output_ids_s[i].extend([1] * (max_len - len(output_ids_s[i])))
        output_attention_mask_s[i].extend([0] * (max_len - len(output_attention_mask_s[i])))
    output_ids_s = torch.tensor(output_ids_s, dtype=torch.int64)
    output_attention_mask_s = torch.tensor(output_attention_mask_s, dtype=torch.int64)
    encoded_outputs_stories = {'input_ids': output_ids_s, 'attention_mask': output_attention_mask_s}
    return encoded_outputs_stories, stories_index


def convert_graph_to_features_update(data, input_event_num=0, use_relation=False, is_eval=False,
                              no_struct=False, use_story_input=False, is_sentence_rl=False, is_otrl_ot=False):
    """
    :param data: 输入的list(dict)格式的数据集
    :param input_event_num: 每一个story预先输出多少个event构成storyline
    :param use_relation: 是否使用temporal prompt
    :param is_eval: 是否进行eval
    :param no_struct: 是否使用structured storyline model
    :param use_story_input: 是否改变采用story_input
    :is_split: 是否对input_id和label进行拆分
    :return:
    """
    # sep token指的是用于分隔的token，注意这里的sep token前后是加了两个空格的,这里的sep token额外加了两个空格
    sep_token, eoe_token, mask_token = ' ; ', '<eoe>', '<mask>'

    def make_event_input(event, mask=False):
        '''把每一个example中的story做成mask标签的格式'''
        if not mask:
            trigger, a1, a2 = event[0].replace(';', ','), event[1].replace(';', ','), event[2].replace(';', ',')
            return "%s%s%s%s%s" % (trigger, sep_token, a1, sep_token, a2)
        else:
            return "%s%s%s%s%s" % (mask_token, sep_token, mask_token, sep_token, mask_token)

    def get_unmask_events(N, num):
        return [[0] + random.Random(n).sample(range(1, 5), num) for n in range(N)]

    samples = []
    counter = 0    # 记录当前处理到第几个样本

    unmask_events = get_unmask_events(len(data), input_event_num)
    assert len(unmask_events) == len(data)

    for v, umask in zip(data, unmask_events):
        # input = [v['title'], eoe_token]    # 是否要加入主题的信息
        input = []
        for i, (e, r) in enumerate(zip(v['events'], v['relations'] + ['eoe'])):
            if i > 0 and no_struct:
                break
            mask = False if i in umask else True
            r_mask = '<%s>' % r
            if use_relation:
                input += [make_event_input(e, mask).lower() + r_mask.lower()]
            else:
                input += [make_event_input(e, mask).lower() + eoe_token.lower()]

        output = []
        for e, r in zip(v['events'], v['relations'] + ['eoe']):
            if not no_struct:
                output += [make_event_input(e).lower() + '<%s>' % r.lower()]
            else:
                output += [make_event_input(e).lower() + eoe_token.lower()]

        # 按照不同的优化情况进行不同的分类别处理方式
        if is_sentence_rl and not is_otrl_ot:
            sample = {'inputs': ''.join(input).lower(),
                      'labels': output}
        elif is_sentence_rl and is_otrl_ot:
            sample = {'inputs': input,
                      'labels': output}
        else:
            sample = {'inputs': ''.join(input).lower(),
                      'labels': ''.join(output).lower()}

        # 这里改了，但是原有要输出的sample['story']和sample['story_input']都还在，增加了sample['split_story']作为它的输出
        # 此外将split_story，story，story_input中的一切都变成小写，这是之前没有实现的,之前的story_input是没有实现小写这项功能的。
        if is_eval:
            keys = ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
            story = []
            for key in keys:
                story.append(v[key].lower())
            sample['split_story'] = story
            sample['story'] = ' '.join(story)

        if use_story_input:
            # 用nltk对其进行分句子，是为了保持前后的一致性
            sample['story_input'] = nltk.sent_tokenize(sample['story'])[0]

        # check some example data
        if counter < 1:
            # 打印一个样本的样例
            logger.info(sample)

        counter += 1
        samples.append(sample)

    return samples


def convert_feature_to_index(all_inputs, tokenizer):
    """根据输入的inputs切分成对应的inputs输入和相应的index"""
    # 存放输入总索引的列表
    all_inputs_index = []
    # 存放输入input id的列表
    all_inputs_ids = []
    # 存放输入attention mask的列表
    all_attention_mask = []
    # 记录当前编码后的最大长度
    max_len = 0
    # 取出每一个story
    for sample_input in all_inputs:
        sample_input_index = []
        sample_input_token = []
        sample_attention_mask = []
        # 取出story中的每一个sentence
        for sentence_item in sample_input:
            # 先对每个句子进行分词
            sentence_item_token = tokenizer.tokenize(sentence_item)
            # 分词的结果并入相应的列表
            sample_input_token.extend(sentence_item_token)
            # 由于一开始就少了一个<s>，注意这里缺少的是0的开头token，因此不用+1，正好索引是能够完全对得上的
            sample_input_index.append(len(sample_input_token))
            # 加入相应长度的mask,1表示不对当前的input id进行mask
            for j in range(len(sentence_item_token)):
                sample_attention_mask.append(1)
        # 转换成对应的id
        sample_input_token = tokenizer.convert_tokens_to_ids(sample_input_token)
        # 插入起始符
        sample_input_token.insert(0, 0)
        # 对起始符号也进行mask
        sample_attention_mask.insert(0, 1)
        # 加入结尾的标签符</s>，因为默认的tokenize原本是会自动加入的，虽然这里没有加入
        sample_input_token.insert(len(sample_input_token), 2)
        # 对结尾的标签也会进行attention的运算
        sample_attention_mask.insert(len(sample_attention_mask), 1)
        # 默认它们的长度应该一致
        assert len(sample_input_token) == len(sample_attention_mask)
        # 记录input id的最大长度
        if len(sample_input_token) > max_len:
            max_len = len(sample_input_token)
        # 加入输入的input id
        all_inputs_ids.append(sample_input_token)
        # 加入输入的attention mask
        all_attention_mask.append(sample_attention_mask)
        # 加入当前样本的索引位置
        all_inputs_index.append(sample_input_index)
    # 按照最长长度对样本进行长度补齐
    for i in range(len(all_inputs_ids)):
        # 补1代表<padding>
        all_inputs_ids[i].extend([1] * (max_len - len(all_inputs_ids[i])))
        # 补0代表对其进行mask，不参与注意力机制的运算
        all_attention_mask[i].extend([0] * (max_len - len(all_attention_mask[i])))
    # 补全成了对应的tensor，方便模型进行运算
    all_inputs_ids = torch.tensor(all_inputs_ids, dtype=torch.int64)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.int64)
    encoded_story = {'input_ids': all_inputs_ids, 'attention_mask': all_attention_mask}
    return encoded_story, all_inputs_index

def convert_event_script_to_input_update_try(data, input_event_num=0, use_relation=False, is_eval=False,
                                        no_struct=False, use_story_input=False, is_rl_sentence=False, is_otrl_ot=False):
    # 一种全新的模版输入格式
    """
    :param data: 输入的list(dict)格式的数据集
    :param input_event_num: 每一个story预先输出多少个event构成storyline
    :param use_relation: 是否使用temporal prompt
    :param is_eval: 是否进行eval
    :param no_struct: 是否使用structured storyline model
    :param use_story_input: 是否改变采用story_input
    :return:
    """   # sep token指的是用于分隔的token，注意这里的sep token前后是加了两个空格的,这里的sep token额外加了两个空格
    # sep token可以考虑加空格或者是不加空格
    sep_token, mask_token = '<eoe>', '<mask>'

    def make_event_input(event, mask=False):
        '''把每一个example中的story做成mask标签的格式'''
        if not mask:
            trigger, a1, a2 = event[0].replace(';', ','), event[1].replace(';', ','), event[2].replace(';', ',')
            return "%s %s %s" % (a1, trigger, a2)
        else:
            return "%s" % (mask_token)

    def get_unmask_events(N, num):
        return [[0] + random.Random(n).sample(range(1, 5), num) for n in range(N)]

    samples = []
    counter = 0    # 记录当前处理到第几个样本

    unmask_events = get_unmask_events(len(data), input_event_num)
    assert len(unmask_events) == len(data)

    for v, umask in zip(data, unmask_events):
        # input = [v['title'], eoe_token]    # 是否要加入主题的信息
        input = []
        for i, (e, r) in enumerate(zip(v['events'], v['relations'] + ['eoe'])):
            if i > 0 and no_struct:
                break
            mask = False if i in umask else True
            r_mask = '<%s>' % r
            # 改了这里，让它第一个输入的就是整个句子，去预测后面的事件脚本
            if i == 0:
                input += [v['sentence1'].lower() + r_mask.lower()]
            else:
                if use_relation:
                    input += [make_event_input(e, mask).lower() + r_mask.lower()]
                else:
                    # 用了新模版的话，不用使用这个
                    input += [make_event_input(e, mask).lower() + eoe_token.lower()]

        output = []
        for i, (e, r) in enumerate(zip(v['events'], v['relations'] + ['eoe'])):
            if i == 0:
                output += [v['sentence1'].lower() + '<%s>' % r.lower()]
            else:
                if not no_struct:
                    output += [make_event_input(e).lower() + '<%s>' % r.lower()]
                else:
                    output += [make_event_input(e).lower() + eoe_token.lower()]

        if is_rl_sentence and not is_otrl_ot:
            sample = {'inputs': ''.join(input).lower(),
                      'labels': output}
        elif is_rl_sentence and is_otrl_ot:
            sample = {'inputs': input,
                      'labels': output}
        else:
            sample = {'inputs': ''.join(input).lower(),
                      'labels': ''.join(output).lower()}

        # 这里改了，但是原有要输出的sample['story']和sample['story_input']都还在，增加了sample['split_story']作为它的输出
        # 此外将split_story，story，story_input中的一切都变成小写，这是之前没有实现的,之前的story_input是没有实现小写这项功能的。
        if is_eval:
            keys = ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
            story = []
            for key in keys:
                story.append(v[key].lower())
            sample['split_story'] = story
            sample['story'] = ' '.join(story)

        if use_story_input:
            # 唯一不同的是这里规划好了不再需要进行分词,change here
            sample['story_input'] = nltk.sent_tokenize(sample['story'])[0].lower()
            # 这里将输入的mask
            # sample['inputs'] = sample['story_input'] + sample['inputs']

        # check some example data
        if counter < 1:
            # 打印一个样本的样例
            logger.info(sample)

        counter += 1
        samples.append(sample)

    return samples


def convert_event_script_to_input_update(data, input_event_num=0, use_relation=False, is_eval=False,
                                        no_struct=False, use_story_input=False, is_rl_sentence=False, is_otrl_ot=False):
    # 一种全新的模版输入格式
    """
    :param data: 输入的list(dict)格式的数据集
    :param input_event_num: 每一个story预先输出多少个event构成storyline
    :param use_relation: 是否使用temporal prompt
    :param is_eval: 是否进行eval
    :param no_struct: 是否使用structured storyline model
    :param use_story_input: 是否改变采用story_input
    :return:
    """   # sep token指的是用于分隔的token，注意这里的sep token前后是加了两个空格的,这里的sep token额外加了两个空格
    # sep token可以考虑加空格或者是不加空格
    sep_token, mask_token = '<eoe>', '<mask>'

    def make_event_input(event, mask=False):
        '''把每一个example中的story做成mask标签的格式'''
        if not mask:
            trigger, a1, a2 = event[0].replace(';', ','), event[1].replace(';', ','), event[2].replace(';', ',')
            return "%s %s %s" % (a1, trigger, a2)
        else:
            return "%s" % (mask_token)

    def get_unmask_events(N, num):
        return [[0] + random.Random(n).sample(range(1, 5), num) for n in range(N)]

    samples = []
    counter = 0    # 记录当前处理到第几个样本

    unmask_events = get_unmask_events(len(data), input_event_num)
    assert len(unmask_events) == len(data)

    for v, umask in zip(data, unmask_events):
        # input = [v['title'], eoe_token]    # 是否要加入主题的信息
        input = []
        # 记录当前的这个样本里应该包含了几件事
        event_count = 0
        for i, (e, r) in enumerate(zip(v['events'], v['relations'] + ['eoe'])):
            if i > 0 and no_struct:
                break
            mask = False if i in umask else True
            r_mask = '<%s>' % r
            if use_relation:
                input += [make_event_input(e, mask).lower() + r_mask.lower()]
                event_count += 1
            else:
                input += [make_event_input(e, mask).lower() + eoe_token.lower()]
                event_count += 1

        output = []
        for e, r in zip(v['events'], v['relations'] + ['eoe']):
            if not no_struct:
                output += [make_event_input(e).lower() + '<%s>' % r.lower()]
            else:
                output += [make_event_input(e).lower() + eoe_token.lower()]

        if is_rl_sentence and not is_otrl_ot:
            sample = {'inputs': ''.join(input).lower(),
                      'labels': output}
        elif is_rl_sentence and is_otrl_ot:
            sample = {'inputs': input,
                      'labels': output}
        else:
            sample = {'inputs': ''.join(input).lower(),
                      'labels': ''.join(output).lower()}
        sample['event_nums'] = event_count
        # 这里改了，但是原有要输出的sample['story']和sample['story_input']都还在，增加了sample['split_story']作为它的输出
        # 此外将split_story，story，story_input中的一切都变成小写，这是之前没有实现的,之前的story_input是没有实现小写这项功能的。
        if is_eval:
            keys = ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
            story = []
            for key in keys:
                story.append(v[key].lower())
            sample['split_story'] = story
            sample['story'] = ' '.join(story)

        if use_story_input:
            # 唯一不同的是这里规划好了不再需要进行分词,change here
            sample['story_input'] = nltk.sent_tokenize(sample['story'])[0].lower()

        # check some example data
        if counter < 1:
            # 打印一个样本的样例
            logger.info(sample)

        counter += 1
        samples.append(sample)

    return samples

def convert_event_script_to_input(data, input_event_num=0, use_relation=False, is_eval=False,
                                        no_struct=False, use_story_input=False):
    # 这个是在测试集在需要使用的
    """
    :param data: 输入的list(dict)格式的数据集
    :param input_event_num: 每一个story预先输出多少个event构成storyline
    :param use_relation: 是否使用temporal prompt
    :param is_eval: 是否进行eval
    :param no_struct: 是否使用structured storyline model
    :param use_story_input: 是否改变采用story_input
    :return:
    """   # sep token指的是用于分隔的token，注意这里的sep token前后是加了两个空格的,这里的sep token额外加了两个空格
    # sep token可以考虑加空格或者是不加空格
    sep_token, mask_token = '<eoe>', '<mask>'

    def make_event_input(event, mask=False):
        '''把每一个example中的story做成mask标签的格式'''
        if not mask:
            trigger, a1, a2 = event[0].replace(';', ','), event[1].replace(';', ','), event[2].replace(';', ',')
            return "%s %s %s" % (a1, trigger, a2)
        else:
            return "%s" % (mask_token)

    def get_unmask_events(N, num):
        return [[0] + random.Random(n).sample(range(1, 5), num) for n in range(N)]

    samples = []
    counter = 0    # 记录当前处理到第几个样本

    unmask_events = get_unmask_events(len(data), input_event_num)
    assert len(unmask_events) == len(data)
    for v, umask in zip(data, unmask_events):
        # input = [v['title'], eoe_token]    # 是否要加入主题的信息
        input = []
        event_count = 0
        for i, (e, r) in enumerate(zip(v['events'], v['relations'] + ['eoe'])):
            if i > 0 and no_struct:
                break
            mask = False if i in umask else True
            r_mask = '<%s>' % r
            if use_relation:
                input += [make_event_input(e, mask).lower() + r_mask.lower()]
                event_count += 1
            else:
                # 用了新模版的话，不用使用这个
                input += [make_event_input(e, mask).lower() + eoe_token.lower()]
                event_count += 1
        output = []
        for e, r in zip(v['events'], v['relations'] + ['eoe']):
            if not no_struct:
                output += [make_event_input(e).lower() + '<%s>' % r.lower()]
            else:
                output += [make_event_input(e).lower() + eoe_token.lower()]

        sample = {'inputs': ''.join(input).lower(),
                  'labels': ''.join(output).lower(),
                  'event_nums': event_count}
        # 这里改了，但是原有要输出的sample['story']和sample['story_input']都还在，增加了sample['split_story']作为它的输出
        # 此外将split_story，story，story_input中的一切都变成小写，这是之前没有实现的,之前的story_input是没有实现小写这项功能的。
        if is_eval:
            sample['story'] = v['story'].lower()

        if use_story_input:
            # 唯一不同的是这里规划好了不再需要进行分词,change here
            sample['story_input'] = nltk.sent_tokenize(sample['story'])[0].lower()
            # 这里将输入的mask
            # sample['inputs'] = sample['story_input'] + sample['inputs']

        # check some example data
        if counter < 1:
            # 打印一个样本的样例
            logger.info(sample)

        counter += 1
        samples.append(sample)

    return samples

def convert_to_event_script_wp_features(data, input_event_num=0, use_relation=False, is_eval=False,
                                               no_struct=False, use_story_input=False):
    eoe_token, mask_token = "<eoe>", "<mask>"

    def make_event_input(event, mask=False):
        if not mask:
            trigger, a1, a2 = event[0].replace(';', ','), event[1].replace(';', ','), event[2].replace(';', ',')
            return "%s %s %s" % (a1, trigger, a2)
        else:
            return "%s" % mask_token

    samples = []
    counter = 0

    for v in data:
        if '[ wp ]' not in v['prompt'].lower():
            input = ['[ wp ]' + v['prompt'] + eoe_token]
        else:
            input = [v['prompt'] + eoe_token]

        for i, (e, r) in enumerate(zip(v['events'], v['relations'] + ['eoe'])):
            if i >= input_event_num and no_struct: break
            mask = True if i >= input_event_num else False

            r_mask = '<%s>' % r
            if use_relation:
                input += [make_event_input(e, mask), r_mask]
            else:
                input += [make_event_input(e, mask), eoe_token]

        output = []
        for e, r in zip(v['events'], v['relations'] + ['eoe']):
            if not no_struct:
                output += [make_event_input(e) + '<%s>' % r]
            else:
                output += [make_event_input(e) + eoe_token]

        sample = {'inputs': ''.join(input).lower(),
                  'labels': ''.join(output).lower()}

        if is_eval:
            sample['story'] = ' '.join(v['story']).lower()

        if use_story_input:
            if input_event_num == 1000:
                sample['labels'] = ' '.join(v['story']).lower()
            else:
                sample['story_input'] = v['prompt'].lower()

        if counter < 1:
            logger.info(sample)

        counter +=  1
        samples.append(sample)

    return samples

def convert_to_event_script_wp_features_update(data, input_event_num=0, use_relation=False, is_eval=False,
                                               no_struct=False, use_story_input=False, is_rl_sentence=False,
                                               is_otrl_ot=False):
    eoe_token, mask_token = "<eoe>", "<mask>"

    def make_event_input(event, mask=False):
        if not mask:
            trigger, a1, a2 = event[0].replace(';', ','), event[1].replace(';', ','), event[2].replace(';', ',')
            return "%s %s %s" % (a1, trigger, a2)
        else:
            return "%s" % mask_token

    samples = []
    counter = 0

    for v in data:
        if '[ wp ]' not in v['prompt'].lower():
            input = ['[ wp ]' + v['prompt'].lower() + eoe_token.lower()]
        else:
            input = [v['prompt'].lower() + eoe_token.lower()]

        for i, (e, r) in enumerate(zip(v['events'], v['relations'] + ['eoe'])):
            if i >= input_event_num and no_struct: break
            mask = True if i >= input_event_num else False

            r_mask = '<%s>' % r
            if use_relation:
                input += [make_event_input(e, mask).lower() + r_mask.lower()]
            else:
                input += [make_event_input(e, mask).lower() + eoe_token.lower()]

        output = []
        for e, r in zip(v['events'], v['relations'] + ['eoe']):
            if not no_struct:
                output += [make_event_input(e).lower() + '<%s>' % r.lower()]
            else:
                output += [make_event_input(e).lower() + eoe_token.lower()]
        if is_rl_sentence and not is_otrl_ot:
            sample = {'inputs': ''.join(input).lower(),
                      'labels': output}
        elif is_rl_sentence and is_otrl_ot:
            sample = {'inputs': input,
                      'labels': output}
        else:
            sample = {'inputs': ''.join(input).lower(),
                      'labels': ''.join(output).lower()}

        if is_eval:
            sample['story'] = ' '.join(v['story']).lower()
            sample['split_story'] = v['story']

        if use_story_input:
            if input_event_num == 1000:
                sample['labels'] = ' '.join(v['story']).lower()
            else:
                sample['story_input'] = v['prompt'].lower()

        if counter < 1:
            logger.info(sample)

        counter +=  1
        samples.append(sample)

    return samples

def convert_to_wp_features_gpt(data, input_event_num=0, use_relation=False, is_eval=False,
                           no_struct=False, use_story_input=False):

    samples = []
    counter = 0
    for v in data:
        sample = {}

        if is_eval:
            sample['story'] = ' '.join(v['story']).lower()
            sample['story_input'] = v['prompt'].lower()

        # check some example data
        if counter < 1:
            logger.info(sample)

        counter += 1
        samples.append(sample)

    return samples


def convert_to_wp_features(data, input_event_num=0, use_relation=False, is_eval=False,
                           no_struct=False, use_story_input=False):

    sep_token, eoe_token, mask_token = ' ; ', '<eoe>', '<mask>'

    def make_event_input(event, mask=False):
        if not mask:
            trigger, a1, a2 = event[0].replace(';', ','), event[1].replace(';', ','), event[2].replace(';', ',')
            return "%s%s%s%s%s" % (trigger, sep_token, a1, sep_token, a2)
        else:
            return "%s%s%s%s%s" % (mask_token, sep_token, mask_token, sep_token, mask_token)

    samples = []
    counter = 0

    for v in data:
        if '[ wp ]' not in v['prompt'].lower():
            input = ['[ wp ] ' + v['prompt'], eoe_token]
        else:
            input = [v['prompt'], eoe_token]

        for i, (e, r) in enumerate(zip(v['events'], v['relations'] + ['eoe'])):
            if i >= input_event_num and no_struct: break
            mask = True if i >= input_event_num else False

            r_mask = '<%s>' % r
            if use_relation:
                input += [make_event_input(e, mask), r_mask]
            else:
                input += [make_event_input(e, mask), eoe_token]

        output = []
        for e, r in zip(v['events'], v['relations'] + ['eoe']):
            if not no_struct:
                output += [make_event_input(e), '<%s>' % r]
            else:
                output += [make_event_input(e), eoe_token]

        sample = {'inputs': ''.join(input).lower(),
                  'labels': ''.join(output).lower()}

        if is_eval:
            sample['story'] = ' '.join(v['story']).lower()

        if use_story_input:
            if input_event_num == 1000:
                sample['labels'] = ' '.join(v['story']).lower()
            else:
                sample['story_input'] = v['prompt'].lower()

        # check some example data
        if counter < 1:
            logger.info(sample)

        counter += 1
        samples.append(sample)

    return samples


# 对输入进行句子切分
def sep_output_sents(output):
    return nltk.tokenize.sent_tokenize(output)


def sep_output_storyline(preds, golds, tokenizer, total=5):
    pred_sl = tokenizer.batch_decode(preds, skip_special_tokens=True)
    gold_sl = tokenizer.batch_decode(golds, skip_special_tokens=True)

    factors = []
    for pred, gold in zip(pred_sl, gold_sl):
        p = re.split('<before>|<after>|<eoe>|<vague>', pred)[:-1]
        g = re.split('<before>|<after>|<eoe>|<vague>', gold)[:-1]
        pairs = zip([pp.split(" ; ")[0] for pp in p], [gg.split(" ; ")[0] for gg in g])
        factor = 1.0 + float(total - sum([x == y for x, y in pairs])) / total
        factors.append(factor)

    return factors

def make_new_story_input_try(pred_storylines, gold_storylines, instance_indices, thresh=0.0):
    all_ids = []
    max_len = 0

    use_pred = torch.ones(len(instance_indices), device=gold_storylines.device)
    for b, i in enumerate(instance_indices):  # 取出组成当前batch的对应的样本id
        # 设定随机种子，判定要采用什么样的故事线
        random.seed(i)
        p = np.random.rand(1)[0]
        # 这里的p用的是0，因此不可能使用ground truth，这里的黄金故事线也要把开头的0去掉
        if p < thresh:
            # 如果随机抽取的数小于thresh，则要使用ground truth作为它的输入
            new_ids = gold_storylines[b][1:]
            # 标记第几个使用了ground truth作为它的故事线
            use_pred[b] = 0.0
        else:
            # 把构成的新的故事线和原来的拼在一起，这样才是完整的输入，因为generate的结构本身就不包含它的输入
            new_ids = pred_storylines[b]

        if len(new_ids) > max_len:  # 新的输入
            max_len = len(new_ids)
        all_ids.append(new_ids)  # 生成新的与之对应的故事线

    new_storylines = []
    for b, ids in enumerate(all_ids):
        if len(ids) < max_len:
            # 对新的故事线设置padding的位置
            pad_ids = torch.ones(max_len - len(ids), dtype=torch.long).to(gold_storylines.device)
            ids = torch.cat([ids, pad_ids])
        new_storylines.append(ids)

    return torch.stack(new_storylines), use_pred


def make_new_story_input(pred_storylines, gold_storylines, tokenizer, story_inputs, instance_indices, thresh=0.0,
                         use_pred_mask=None):
    all_ids = []
    max_len = 0

    for b, i in enumerate(instance_indices):   # 取出组成当前batch的对应的样本id
        # 加了一个<eoe>作为它的结束分隔符，加了一个[0]作为它的起始符号<s>,因为只用tokenizer.tokenize()的方法，本身不会在开头加上为0的token_id
        ids = [0] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(story_inputs[i]) + ['<eoe>'])
        # 把id都转换成对应的LongTensor
        new_ids = torch.LongTensor(ids).to(gold_storylines.device)

        # 设定随机种子，判定要采用什么样的故事线
        random.seed(i)
        p = np.random.rand(1)[0]
        # 这里的p用的是0，因此不可能使用ground truth，这里的黄金故事线也要把开头的0去掉
        if p < thresh:
            # 如果随机抽取的数小于thresh，则要使用ground truth作为它的输入
            new_ids = torch.cat([new_ids, gold_storylines[b][1:]])
            # 标记第几个使用了ground truth作为它的故事线
            use_pred_mask[b] = 0
        else:
            # 把构成的新的故事线和原来的拼在一起，这样才是完整的输入，因为generate的结构本身就不包含它的输入
            new_ids = torch.cat([new_ids, pred_storylines[b]])

        if len(new_ids) > max_len:  # 新的输入
            max_len = len(new_ids)
        all_ids.append(new_ids)    # 生成新的与之对应的故事线

    new_storylines = []
    for b, ids in enumerate(all_ids):
        if len(ids) < max_len:
            # 对新的故事线设置padding的位置
            pad_ids = torch.ones(max_len - len(ids), dtype=torch.long).to(gold_storylines.device)
            ids = torch.cat([ids, pad_ids])
        new_storylines.append(ids)
    if use_pred_mask != None:
        return torch.stack(new_storylines), use_pred_mask

    return torch.stack(new_storylines)

def nullify_story_inputs(stories, tokenizer, story_inputs, instance_indices):
    """这里不是很懂是什么意思"""
    """
    stories: 指的是story的groundtruth
    tokenizer: 初始化的分词器
    story_inputs: 故事线的真是输入
    instance_indices: 当前的batch中包含的对应样本的id
    """
    for b, i in enumerate(instance_indices):
        temp = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(story_inputs[i]))
        # 注意这里跳过了第一位，是为了保证第一位的输入永远都是标签<s>，符合Bart模型的需求
        # 这里之所以要+1是因为上面用的是tokenizer.convert_tokens_to_ids，不会自动的在头部加<s>，在尾部加</s>，所以把最开头的</s>标签也要算上
        for j in range(len(temp) + 1):
            # 相当于这里对第一个句子并不会计算它的损失函数
            stories[b, j] = -100
    return stories


def convert_story_graph_to_features(data, use_relation, input_event_num=5, use_story_input=False):

    sep_token, eoe_token = ' ; ', '<eoe>'

    def make_event_input(event):
        trigger, a1, a2 = event[0].replace(';', ','), event[1].replace(';', ','), event[2].replace(';', ',')
        return "%s%s%s%s%s" % (trigger, sep_token, a1, sep_token, a2)

    samples = []
    counter = 0

    for v in data:
        #input = [v['title'], eoe_token]
        input = []
        upper = min(4, input_event_num)
        for e, r in zip(v['events'][:upper], v['relations'][:upper]):
            r_mask = '<%s>' % r
            if use_relation:
                input += [make_event_input(e), r_mask]
            else:
                input += [make_event_input(e), eoe_token]
        if input_event_num == 5:
            input += [make_event_input(v['events'][-1]), eoe_token]

        output = v['story'].lower()

        if use_story_input:
            input = [nltk.tokenize.sent_tokenize(output)[0], '<eoe>'] + input

        sample = {'inputs': ''.join(input).lower(),
                  'labels': output.lower()}

        counter += 1
        # check some example data
        if counter < 3:
            logger.info(sample)

        samples.append(sample)

    return samples


def convert_story_to_features(stories, tokenizer, max_seq_length=128, eval=False):

    samples = []
    counter, global_max = 0, 0

    for story in stories:

        mask_ids = []

        new_tokens = [tokenizer.bos_token]
        orig_to_tok_map = []

        mask_ids.append(1)
        for i, token in enumerate(story['story_input']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
            mask_ids += [1]*len(temp_tokens)

        new_tokens += [tokenizer.eos_token]
        mask_ids += [1]

        length = len(new_tokens)

        if length > global_max:
            global_max = length

        # max_seq_length set to be global max
        if len(new_tokens) > max_seq_length:
            logger.info("%s exceeds max length!!!" % len(new_tokens))

        # padding
        new_tokens += [tokenizer.pad_token] * (max_seq_length - len(new_tokens))
        mask_ids += [0] * (max_seq_length - len(mask_ids))

        input_ids = tokenizer.convert_tokens_to_ids(new_tokens)
        assert len(input_ids) == len(mask_ids)

        index_starts, index_ends = [], []
        event_rel_seq = ""
        for i, (t, a1, a2) in enumerate(zip(story['triggers'], story['arg1'], story['arg2'])):
            # trigger must exist
            index_starts.append(orig_to_tok_map[t[0]])
            index_ends.append(orig_to_tok_map[t[-1]])
            event_rel_seq += "%s; " % ' '.join(story['story_input'][t[0]:t[-1]+1])
            if a1:
                index_starts.append(orig_to_tok_map[a1[0]])
                index_ends.append(orig_to_tok_map[a1[-1]])
                event_rel_seq += "%s; " % ' '.join(story['story_input'][a1[0]:a1[-1] + 1])
            else:
                # if argument doesn't exist use the eos tok to replace
                index_starts.append(-1)
                index_ends.append(-1)
                event_rel_seq += "; "
            if a2:
                index_starts.append(orig_to_tok_map[a2[0]])
                index_ends.append(orig_to_tok_map[a2[-1]])
                event_rel_seq += "%s; " % ' '.join(story['story_input'][a2[0]:a2[-1] + 1])
            else:
                # if argument doesn't exist use the eos tok to replace
                index_starts.append(-1)
                index_ends.append(-1)
                event_rel_seq += "; "
            event_rel_seq += "%s; " % story['relations'][i]

        assert len(index_starts) == len(index_ends) == 3 * len(story['triggers'])

        sample = {'input_ids': input_ids,
                  'attention_mask': mask_ids,
                  'index_starts': index_starts,
                  'index_ends': index_ends,
                  'relations': [rel_to_index[r.lower()] for r in story['relations']],
                  'labels': ' '.join(story['story_output']).lower()}

        if eval:
            sample['story_input'] = ' '.join(story['story_input']).lower()
            sample['event_rel_seq'] = event_rel_seq

        counter += 1

        # check some example data
        if counter < 1:
            print(story)
            print(sample)

        samples.append(sample)

    logger.info("Global Max Length is %s !!!" % global_max)

    return samples


def select_field(data, field):
    return [ex[field] for ex in data]


class ClassificationReport:
    def __init__(self, name, true_labels: List[Union[int, str]],
                 pred_labels: List[Union[int, str]]):

        assert len(true_labels) == len(pred_labels)
        self.name = name
        self.num_tests = len(true_labels)
        self.total_truths = Counter(true_labels)
        self.total_predictions = Counter(pred_labels)
        self.labels = sorted(set(true_labels) | set(pred_labels))
        self.confusion_mat = self.confusion_matrix(true_labels, pred_labels)
        self.accuracy = sum(y == y_ for y, y_ in zip(true_labels, pred_labels)) / len(true_labels)
        self.trim_label_width = 15
        self.rel_f1 = 0.0
        self.res_dict = {}

    @staticmethod
    def confusion_matrix(true_labels: List[str], predicted_labels: List[str]) \
            -> Mapping[str, Mapping[str, int]]:
        mat = defaultdict(lambda: defaultdict(int))
        for truth, prediction in zip(true_labels, predicted_labels):
            mat[truth][prediction] += 1
        return mat

    def __repr__(self):
        res = f'Name: {self.name}\t Created: {datetime.now().isoformat()}\t'
        res += f'Total Labels: {len(self.labels)} \t Total Tests: {self.num_tests}\n'
        display_labels = [label[:self.trim_label_width] for label in self.labels]
        label_widths = [len(l) + 1 for l in display_labels]
        max_label_width = max(label_widths)
        header = [l.ljust(w) for w, l in zip(label_widths, display_labels)]
        header.insert(0, ''.ljust(max_label_width))
        res += ''.join(header) + '\n'
        for true_label, true_disp_label in zip(self.labels, display_labels):
            predictions = self.confusion_mat[true_label]
            row = [true_disp_label.ljust(max_label_width)]
            for pred_label, width in zip(self.labels, label_widths):
                row.append(str(predictions[pred_label]).ljust(width))
            res += ''.join(row) + '\n'
        res += '\n'

        def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else numr / denr

        def num_to_str(num):
            return '0' if num == 0 else str(num) if type(num) is int else f'{num:.4f}'

        n_correct = 0
        n_true = 0
        n_pred = 0

        all_scores = []
        header = ['Total  ', 'Predictions', 'Correct', 'Precision', 'Recall  ', 'F1-Measure']
        res += ''.ljust(max_label_width + 2) + '  '.join(header) + '\n'
        head_width = [len(h) for h in header]

        exclude_list = ['None']
        #if "matres" in self.name: exclude_list.append('VAGUE')

        for label, width, display_label in zip(self.labels, label_widths, display_labels):
            if label not in exclude_list:
                total_count = self.total_truths.get(label, 0)
                pred_count = self.total_predictions.get(label, 0)

                n_true += total_count
                n_pred += pred_count

                correct_count = self.confusion_mat[label][label]
                n_correct += correct_count

                precision = safe_division(correct_count, pred_count)
                recall = safe_division(correct_count, total_count)
                f1_score = safe_division(2 * precision * recall, precision + recall)
                all_scores.append((precision, recall, f1_score))
                self.res_dict[label] = (f1_score, total_count)
                row = [total_count, pred_count, correct_count, precision, recall, f1_score]
                row = [num_to_str(cell).ljust(w) for cell, w in zip(row, head_width)]
                row.insert(0, display_label.rjust(max_label_width))
                res += '  '.join(row) + '\n'

        # weighing by the truth label's frequency
        label_weights = [safe_division(self.total_truths.get(label, 0), self.num_tests)
                         for label in self.labels if label not in exclude_list]
        weighted_scores = [(w * p, w * r, w * f) for w, (p, r, f) in zip(label_weights, all_scores)]

        assert len(label_weights) == len(weighted_scores)

        res += '\n'
        res += '  '.join(['Weighted Avg'.rjust(max_label_width),
                          ''.ljust(head_width[0]),
                          ''.ljust(head_width[1]),
                          ''.ljust(head_width[2]),
                          num_to_str(sum(p for p, _, _ in weighted_scores)).ljust(head_width[3]),
                          num_to_str(sum(r for _, r, _ in weighted_scores)).ljust(head_width[4]),
                          num_to_str(sum(f for _, _, f in weighted_scores)).ljust(head_width[5])])

        print(n_correct, n_pred, n_true)

        precision = safe_division(n_correct, n_pred)
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2.0 * precision * recall, precision + recall)

        res += f'\n Total Examples: {self.num_tests}'
        res += f'\n Overall Precision: {num_to_str(precision)}'
        res += f'\n Overall Recall: {num_to_str(recall)}'
        res += f'\n Overall F1: {num_to_str(f1_score)} '
        self.rel_f1 = f1_score
        return res


def get_template_arguments(template):
    num_arg_toks = sum([x.split('-')[-1][:3] == 'ARG' and x.split('-')[-1][-1].isdigit() for x in template[3]])
    return num_arg_toks


# TODO: Find the most dominant verb in the sentence!
def get_medium_template(templates):
    # sort templates with the following order: fewer arguments < more arguments < no arguments
    temps = sorted([(get_template_arguments(template), i) for i, template in enumerate(templates)])
    idx = (len(temps) - 1) // 2
    return templates[temps[idx][1]]


def merge_dfs(df1, df2, df3, has_gold=False):
    df = pd.merge(pd.merge(df1, df2, on='PassageKey'), df3, on='PassageKey')
    if has_gold:
        df.columns = ['PassageKey', 'Prediction_x', 'Gold_x', 'Prediction_y', 'Gold_y', 'Prediction_z', 'Gold_z']
    else:
        df.columns = ['PassageKey', 'Prediction_x', 'Prediction_y', 'Prediction_z']
    return df


def majority(row):
    preds = mode([row['Prediction_x'], row['Prediction_y'], row['Prediction_z']])
    return preds[0][0] if preds[1][0] >= 2 else 'VAGUE'


def strict_count(row):
    preds = mode([row['Prediction_x'], row['Prediction_y'], row['Prediction_z']])
    return preds[0][0] if preds[1][0] == 3 else 'VAGUE'


def single_edge_distribution(df):

    df['majority'] = df.apply(lambda row: majority(row), axis=1)
    df['strict'] = df.apply(lambda row: strict_count(row), axis=1)

    print("Single Edge Distribution by majority vote.")
    print(Counter(df['majority'].tolist()))

    print("Single Edge Distribution by consensus.")
    print(Counter(df['strict'].tolist()))
    return df


def get_story_ids(df):
    df['story_id'] = df.apply(lambda row: row['PassageKey'][:-2], axis=1)
    return df


def categorize(x):
    ordering = list(set(x.tolist()))
    if len(ordering) == 1:
        return ordering[0]
    else:
        return 'VAGUE'
