import json
import numpy as np
import random
import copy
from tqdm import tqdm
import torch

def select_field(samples, key):
    return [sample[key] for sample in samples]

def convert_raw_data_to_macro_raw_dataset(raw_data_path, args, logger=None, mode=None):

    duplicated_times = 2
    datas = json.load(open(raw_data_path))
    logger.info("whole data num is: {}".format(len(datas)))
    if mode is not None:
        if mode not in ['eval', 'event_centric', 'event_temporal_mask',
                        'shuffle_event_temporal_mask', 'shuffle_event_mask_no_temporal']:
            logger.info('the mode should be in eval | event_centric | event_temporal_mask'
                        '| shuffle_event_temporal_mask | shuffle_event_mask_no_temporal')

    if mode == 'event_centric':
        samples = []
        for data in tqdm(datas):
            for j in range(duplicated_times):
                sentence_list, temporal_list = data['data'], data['relations']
                mask_index = np.argwhere((np.random.rand(len(sentence_list)) <
                                          args.stage1_event_mask_ratio) == 1).squeeze(1).tolist()
                if not mask_index:
                    mask_index =  random.sample(range(0, len(sentence_list)), 1)
                if len(mask_index) == len(sentence_list):
                    mask_index.pop(random.randint(0, len(sentence_list) - 1))
                encode_input = ''
                for i in range(len(sentence_list)):
                    if i in mask_index:
                        encode_input += '<mask><%s>' % temporal_list[i].lower()
                    else:
                        encode_input += sentence_list[i].lower() + "<%s>" % temporal_list[i].lower()
                decode_input = []
                for i in range(len(sentence_list)):
                    decode_input.append(sentence_list[i].lower())
                    decode_input.append('<%s>' % temporal_list[i].lower())
                sample = {'encode_input': encode_input,
                          'decode_input': decode_input,
                          'mask_index': mask_index}
                samples.append(sample)
        return samples

    elif mode == 'event_temporal_mask':
        samples = []
        for data in tqdm(datas):
            for j in range(duplicated_times):
                sentence_list, temporal_list = data['data'], data['relations']
                mask_index = np.argwhere((np.random.rand(len(sentence_list)) <
                                          args.stage2_event_mask_ratio) == 1).squeeze(1).tolist()
                if not mask_index:
                    mask_index =  random.sample(range(0, len(sentence_list)), 1)
                if len(mask_index) == len(sentence_list):
                    mask_index.pop(random.randint(0, len(sentence_list) - 1))
                encode_input = ''
                for i in range(len(sentence_list)):
                    if i in mask_index:
                        encode_input += '<mask><%s>' % temporal_list[i].lower()
                    else:
                        encode_input += sentence_list[i].lower() + '<mask>'
                decode_input = []
                for i in range(len(sentence_list)):
                    decode_input.append(sentence_list[i].lower())
                    decode_input.append('<%s>' % temporal_list[i].lower())
                sample = {'encode_input': encode_input,
                          'decode_input': decode_input,
                          'mask_index': mask_index}
                samples.append(sample)
        return samples

    elif mode == "shuffle_event_temporal_mask":
        samples = []
        for data in tqdm(datas):
            for j in range(duplicated_times):
                sentence_list, temporal_list = data['data'], data['relations']
                shuffle_index = list(range(len(sentence_list)))
                random.shuffle(shuffle_index)
                mask_index = np.argwhere((np.random.rand(len(sentence_list)) <
                                          args.stage3_event_mask_ratio) == 1).squeeze(1).tolist()
                if not mask_index:
                    mask_index =  random.sample(range(0, len(sentence_list)), 1)
                if len(mask_index) == len(sentence_list):
                    mask_index.pop(random.randint(0, len(sentence_list) - 1))
                encode_input = ''
                for i in range(len(sentence_list)):
                    if i in mask_index:
                        encode_input += '<mask><%s>' % temporal_list[shuffle_index[i]].lower()
                    else:
                        encode_input += sentence_list[shuffle_index[i]].lower() + '<mask>'
                decode_input = ''
                for i in range(len(sentence_list)):
                        decode_input += sentence_list[i].lower() + '<%s>' % temporal_list[i].lower()
                sample = {'encode_input': encode_input,
                          'decode_input': decode_input}
                samples.append(sample)
        return samples

    elif mode == 'shuffle_event_mask_no_temporal':
        samples = []
        for data in tqdm(datas):
            for j in range(duplicated_times):
                sentence_list, temporal_list = data['data'], data['relations']
                shuffle_index = list(range(len(sentence_list)))
                random.shuffle(shuffle_index)
                mask_index = np.argwhere((np.random.rand(len(sentence_list)) <
                                          args.stage4_event_mask_ratio) == 1).squeeze(1).tolist()
                if not mask_index:
                    mask_index = random.sample(range(0, len(sentence_list)), 1)
                if len(mask_index) == len(sentence_list):
                    mask_index.pop(random.randint(0, len(sentence_list) - 1))
                encode_input = ''
                for i in range(len(sentence_list)):
                    if i in mask_index:
                        encode_input += '<mask>%s' % '<mask>'
                    else:
                        encode_input += sentence_list[shuffle_index[i]].lower() + '<mask>'
                decode_input = ''
                for i in range(len(sentence_list)):
                    decode_input += sentence_list[i].lower() + '<%s>' % temporal_list[i].lower()
                sample = {'encode_input': encode_input,
                          'decode_input': decode_input}
                samples.append(sample)
        return samples

    else:
        samples = []
        for data in tqdm(datas):
            sentence_list, temporal_list = data['data'], data['relations']
            encode_input = ''
            for i in range(len(sentence_list)):
                if i == 0:
                    encode_input += sentence_list[i].lower() + '<%s>' % temporal_list[i].lower()
                else:
                    encode_input += '<mask>' + '<%s>' % temporal_list[i].lower()
            decode_input = ''
            for i in range(len(sentence_list)):
                decode_input += sentence_list[i].lower() + '<%s>' % temporal_list[i].lower()
            sample = {'encode_input': encode_input, 'decode_input': decode_input}
            samples.append(sample)
        return samples

def convert_raw_data_to_raw_dataset(raw_data_path, args, logger=None, mode=None):

    duplicated_times = 2
    datas = json.load(open(raw_data_path))
    logger.info("whole data num is: {}".format(len(datas)))
    if mode is not None:
        if mode not in ['eval', 'event_centric', 'event_temporal_mask',
                        'shuffle_event_temporal_mask', 'shuffle_event_mask_no_temporal', "autoregressive_mask"]:
            logger.info('the mode should be in eval | event_centric | event_temporal_mask'
                        '| shuffle_event_temporal_mask | shuffle_event_mask_no_temporal')

    if mode == 'event_centric':
        samples = []
        for data in tqdm(datas):
            for j in range(duplicated_times):
                sentence_list, temporal_list = data['data'], data['relations']
                mask_index = np.argwhere((np.random.rand(len(sentence_list)) <
                                          args.stage1_event_mask_ratio) == 1).squeeze(1).tolist()
                if not mask_index:
                    mask_index =  random.sample(range(0, len(sentence_list)), 1)
                if len(mask_index) == len(sentence_list):
                    mask_index.pop(random.randint(0, len(sentence_list) - 1))
                encode_input = ''
                for i in range(len(sentence_list)):
                    if i in mask_index:
                        encode_input += '<mask><%s>' % temporal_list[i].lower()
                    else:
                        encode_input += sentence_list[i].lower() + "<%s>" % temporal_list[i].lower()
                decode_input = ''
                for i in range(len(sentence_list)):
                        decode_input += sentence_list[i].lower() + "<%s>" % temporal_list[i].lower()
                sample = {'encode_input': encode_input,
                          'decode_input': decode_input}
                samples.append(sample)
        return samples

    elif mode == 'event_temporal_mask':
        samples = []
        for data in tqdm(datas):
            for j in range(duplicated_times):
                sentence_list, temporal_list = data['data'], data['relations']
                # 这里改成随机抽取其中的一个事件
                mask_index = random.sample(range(0, len(sentence_list)), 1)
                if not mask_index:
                    mask_index =  random.sample(range(0, len(sentence_list)), 1)
                if len(mask_index) == len(sentence_list):
                    mask_index.pop(random.randint(0, len(sentence_list) - 1))
                encode_input = ''
                for i in range(len(sentence_list)):
                    if i in mask_index:
                        encode_input += '<mask><%s>' % temporal_list[i].lower()
                    else:
                        encode_input += sentence_list[i].lower() + '<mask>'
                decode_input = ''
                for i in range(len(sentence_list)):
                        decode_input += sentence_list[i].lower() + '<%s>' % temporal_list[i].lower()
                sample = {'encode_input': encode_input,
                          'decode_input': decode_input}
                samples.append(sample)
        return samples

    elif mode == "shuffle_event_temporal_mask":
        samples = []
        for data in tqdm(datas):
            for j in range(duplicated_times):
                sentence_list, temporal_list = data['data'], data['relations']
                shuffle_index = list(range(len(sentence_list)))
                random.shuffle(shuffle_index)
                mask_index = np.argwhere((np.random.rand(len(sentence_list)) <
                                          args.stage3_event_mask_ratio) == 1).squeeze(1).tolist()
                if not mask_index:
                    mask_index =  random.sample(range(0, len(sentence_list)), 1)
                if len(mask_index) == len(sentence_list):
                    mask_index.pop(random.randint(0, len(sentence_list) - 1))
                encode_input = ''
                for i in range(len(sentence_list)):
                    if i in mask_index:
                        encode_input += '<mask><%s>' % temporal_list[shuffle_index[i]].lower()
                    else:
                        encode_input += sentence_list[shuffle_index[i]].lower() + '<mask>'
                decode_input = ''
                for i in range(len(sentence_list)):
                        decode_input += sentence_list[i].lower() + '<%s>' % temporal_list[i].lower()
                sample = {'encode_input': encode_input,
                          'decode_input': decode_input}
                samples.append(sample)
        return samples

    elif mode == 'shuffle_event_mask_no_temporal':
        samples = []
        for data in tqdm(datas):
            for j in range(duplicated_times):
                sentence_list, temporal_list = data['data'], data['relations']
                shuffle_index = list(range(len(sentence_list)))
                random.shuffle(shuffle_index)
                mask_index = np.argwhere((np.random.rand(len(sentence_list)) <
                                          args.stage4_event_mask_ratio) == 1).squeeze(1).tolist()
                if not mask_index:
                    mask_index = random.sample(range(0, len(sentence_list)), 1)
                if len(mask_index) == len(sentence_list):
                    mask_index.pop(random.randint(0, len(sentence_list) - 1))
                encode_input = ''
                for i in range(len(sentence_list)):
                    if i in mask_index:
                        encode_input += '<mask>%s' % '<mask>'
                    else:
                        encode_input += sentence_list[shuffle_index[i]].lower() + '<mask>'
                decode_input = ''
                for i in range(len(sentence_list)):
                    decode_input += sentence_list[i].lower() + '<%s>' % temporal_list[i].lower()
                sample = {'encode_input': encode_input,
                          'decode_input': decode_input}
                samples.append(sample)
        return samples
    elif mode == "autoregressive_mask":
        samples = []
        for data in tqdm(datas):
            sentence_list, temporal_list = data['data'], data['relations']
            mask_index = list(range(1, len(sentence_list)))
            if not mask_index:
                mask_index = random.sample(range(0, len(sentence_list)), 1)
            if len(mask_index) == len(sentence_list):
                mask_index.pop(random.randint(0, len(sentence_list) - 1))
            encode_input = ''
            for i in range(len(sentence_list)):
                if i in mask_index:
                    encode_input += '<mask><%s>' % temporal_list[i].lower()
                else:
                    encode_input += sentence_list[i].lower() + "<%s>" % temporal_list[i].lower()
            decode_input = ''
            for i in range(len(sentence_list)):
                decode_input += sentence_list[i].lower() + "<%s>" % temporal_list[i].lower()
            sample = {'encode_input': encode_input,
                      'decode_input': decode_input}
            samples.append(sample)
        return samples

    else:
        samples = []
        for data in tqdm(datas):
            sentence_list, temporal_list = data['data'], data['relations']
            encode_input = ''
            for i in range(len(sentence_list)):
                if i == 0:
                    encode_input += sentence_list[i].lower() + '<%s>' % temporal_list[i].lower()
                else:
                    encode_input += '<mask>' + '<%s>' % temporal_list[i].lower()
            decode_input = ''
            for i in range(len(sentence_list)):
                decode_input += sentence_list[i].lower() + '<%s>' % temporal_list[i].lower()
            sample = {'encode_input': encode_input, 'decode_input': decode_input}
            samples.append(sample)
        return samples

def convert_dataset_to_tensordataset(dataset, tokenizer):
    encode_input = select_field(dataset, 'encode_input')
    decode_input = select_field(dataset, 'decode_input')
    encode_tokenized = tokenizer(encode_input, padding=True, truncation=True, return_tensors="pt")
    decode_tokenized = tokenizer(decode_input, padding=True, truncation=True, return_tensors="pt")
    encode_input_ids = encode_tokenized['input_ids']
    encode_attention_masks = encode_tokenized['attention_mask']
    decode_attention_masks = decode_tokenized['attention_mask']
    labels = decode_tokenized['input_ids']
    return encode_input_ids, encode_attention_masks, decode_attention_masks, labels

def macro_tokenizer(decode_input, mask_index, tokenizer, temporal_mask=False):
    decode_input_ids = []
    decode_attention_mask = []
    loss_mask = []
    global_max_len = 0
    assert len(decode_input) == len(mask_index)
    for i in tqdm(range(len(decode_input))):
        sample_decode_input = decode_input[i]
        sample_temporal_mask = mask_index[i]
        sample_decode_input_ids = [0]
        sample_decode_attention_mask = [1]
        sample_loss_mask = [0]
        for j in range(len(sample_decode_input) // 2):
            sentence_decode_input = sample_decode_input[2*j:2*(j+1)]
            if j in sample_temporal_mask:
                token_result1 = tokenizer(sentence_decode_input[0])['input_ids'][1:-1]
                sample_decode_input_ids.extend(token_result1)
                sample_decode_attention_mask.extend([1] * len(token_result1))
                sample_loss_mask.extend([0] * len(token_result1))

                token_result2 = tokenizer(sentence_decode_input[1])['input_ids'][1:-1]
                sample_decode_input_ids.extend(token_result2)
                sample_decode_attention_mask.extend([1] * len(token_result2))
                sample_loss_mask.extend([1] * len(token_result2))

            else:
                token_result1 = tokenizer(sentence_decode_input[0])['input_ids'][1:-1]
                sample_decode_input_ids.extend(token_result1)
                sample_decode_attention_mask.extend([1] * len(token_result1))
                sample_loss_mask.extend([1] * len(token_result1))
                if not temporal_mask:
                    token_result2 = tokenizer(sentence_decode_input[1])['input_ids'][1:-1]
                    sample_decode_input_ids.extend(token_result2)
                    sample_decode_attention_mask.extend([1] * len(token_result2))
                    sample_loss_mask.extend([1] * len(token_result2))
                else:
                    token_result2 = tokenizer(sentence_decode_input[1])['input_ids'][1:-1]
                    sample_decode_input_ids.extend(token_result2)
                    sample_decode_attention_mask.extend([1] * len(token_result2))
                    sample_loss_mask.extend([0] * len(token_result2))

        sample_decode_input_ids.append(2)
        sample_decode_attention_mask.append(1)
        sample_loss_mask.append(0)

        assert len(sample_decode_input_ids) == len(sample_decode_attention_mask) == len(sample_loss_mask)
        if len(sample_decode_input_ids) > global_max_len:
            global_max_len = len(sample_decode_input_ids)

        decode_input_ids.append(sample_decode_input_ids)
        decode_attention_mask.append(sample_decode_attention_mask)
        loss_mask.append(sample_loss_mask)

    _ = [sample_decode_input_ids.extend([1] * (global_max_len - len(sample_decode_input_ids)))
                        for sample_decode_input_ids in decode_input_ids]
    _ = [sample_decode_attention_mask.extend([0] * (global_max_len - len(sample_decode_attention_mask)))
                        for sample_decode_attention_mask in decode_attention_mask]
    _ = [sample_loss_mask.extend([1] * (global_max_len - len(sample_loss_mask)))
                        for sample_loss_mask in loss_mask]

    decode_input_ids = torch.tensor(decode_input_ids, dtype=torch.long)
    decode_attention_mask = torch.tensor(decode_attention_mask, dtype=torch.long)
    loss_mask = torch.tensor(loss_mask, dtype=torch.long)

    return {'input_ids': decode_input_ids, 'attention_mask': decode_attention_mask, 'loss_mask': loss_mask}


def convert_dataset_to_macro_tensordataset(dataset, tokenizer, temporal_mask=False):
    encode_input = select_field(dataset, 'encode_input')
    decode_input = select_field(dataset, 'decode_input')
    mask_index = select_field(dataset, 'mask_index')
    decode_tokenizer = macro_tokenizer(decode_input, mask_index, tokenizer, temporal_mask=temporal_mask)
    encode_tokenized = tokenizer(encode_input, padding=True, truncation=True, return_tensors="pt")
    encode_input_ids = encode_tokenized['input_ids']
    encode_attention_masks = encode_tokenized['attention_mask']
    labels = decode_tokenizer['input_ids']
    decode_attention_mask = decode_tokenizer['attention_mask']
    loss_mask = decode_tokenizer['loss_mask']
    return encode_input_ids, encode_attention_masks, decode_attention_mask, labels, loss_mask