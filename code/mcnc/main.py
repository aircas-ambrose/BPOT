import gc
import pickle
import torch
import random
import time
import logging
import math
import datetime
import numpy as np
import os
import argparse
from datetime import date
from models.base.cot import ComplementEntropy
from transformers import RobertaTokenizer,get_linear_schedule_with_warmup,RobertaForMultipleChoice,AdamW,get_scheduler,get_polynomial_decay_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from models.bart_1cls import bart_1cls
from models.bart_mask_random import bart_mask_random
from tools.bart_dataset_random import bart_dataset_random
from multiprocessing import Pool
from torch.optim import Adam
from tools.common import seed_everything,Args,format_time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import get_rank
from torch.utils.tensorboard import SummaryWriter  
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
# from apex import amp
from utils import *

MODEL_CLASSES = {
    'bart_1cls': bart_1cls,
    'bart_mask_random' : bart_mask_random
}

def setup_root_logger(distributed_rank, save_dir ,file_name=None):
    if distributed_rank not in [-1, 0]:
        logger_not_root = logging.getLogger(name=__name__)
        logger_not_root.propagate=False
        return logger_not_root
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    if save_dir:
        formatter = '%(asctime)s-%(levelname)s-%(name)s | %(message)s'
        file_name = file_name
        fn = logging.FileHandler(os.path.join(save_dir, file_name), mode="w")
        fn.setLevel(logging.DEBUG)
        fn.setFormatter(formatter)
        root_logger.addHandler(fn)
    return root_logger

def cal_weighted_loss(loss1, loss2, global_step, args):
    weight1 = (float(global_step) / args.total_steps) * (args.loss_ratio_start - args.loss_ratio_end) + args.loss_ratio_end
    weight2 = (float(global_step) / args.total_steps) * (args.loss_ratio_end - args.loss_ratio_start) + args.loss_ratio_start
    return loss1 * weight2 + loss2 * weight1


def cnt_loss(labels, pad_token_id):

    cnt_labels = labels.ne(pad_token_id).sum(dim=1)
    cnt_labels.require_grads=False
    return cnt_labels

def class_acc(preds, labels):    

    correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()         
    acc = correct.sum().item() / len(correct)
    return acc

def train(args,train_dataloader,model,optimizer,lr_scheduler,writer,logger=None,global_step=0, macro_loss=False):

    if not args.ablation_event and not args.ablation_timing:
        event_centric_train_dataloader, event_temporal_mask_train_dataloader = train_dataloader
    elif args.ablation_event and not args.ablation_timing:
        train_dataloader = train_dataloader
    else:
        train_dataloader = train_dataloader
    t0 = time.time()

    avg_loss = []

    pad_token_id = model.tokenizer.pad_token_id
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    model.zero_grad()
    if not args.ablation_event and not args.ablation_timing:
        for step, (event_centric_batch, event_temporal_mask_batch) in enumerate(zip(event_centric_train_dataloader,
                                                                                    event_temporal_mask_train_dataloader)):
            model.train()

            event_centric_batch = [t.long() for t in event_centric_batch]
            event_temporal_mask_batch = [t.long() for t in event_temporal_mask_batch]

            batch = [torch.cat((sample_event_centric_batch, sample_event_temporal_mask_batch), dim=0)
                     for sample_event_centric_batch, sample_event_temporal_mask_batch in zip(event_centric_batch, event_temporal_mask_batch)]


            batch = tuple(t.to(args.device) for t in batch)
            if not macro_loss:
                labels = batch[-1]
            else:
                labels = batch[-2]

            loss, loss_mask = model(*tuple(batch))
            valid_loss_count = cnt_loss(labels, pad_token_id)
            if macro_loss:
                macro_loss_count = (loss_mask == 0).sum(dim=1)
            split_point = args.per_gpu_train_batch_size
            if macro_loss:
                loss1 = (loss[:split_point, :] * loss_mask[:split_point, :]).sum() / \
                        sum(valid_loss_count[:split_point] - macro_loss_count[:split_point]) \
                        + (loss[:split_point, :] * (1 - loss_mask[:split_point, :])).sum() / sum(macro_loss_count[:split_point])

                loss2 = (loss[split_point:, :] * loss_mask[split_point:, :]).sum() / \
                        sum(valid_loss_count[split_point:] - macro_loss_count[split_point:]) \
                        + (loss[split_point:, :] * (1 - loss_mask[split_point:, :])).sum() / sum(macro_loss_count[split_point:])
            else:

                loss1 = loss[:split_point, :].sum() / sum(valid_loss_count[:split_point])
                loss2 = loss[split_point:, :].sum() / sum(valid_loss_count[split_point:])
            loss = cal_weighted_loss(loss1, loss2, global_step, args)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:

                pass
            else:

                loss.backward()

            loss = loss.item()
            avg_loss.append(loss)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0*args.gradient_accumulation_steps)
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()
                global_step += 1

            if global_step % 500==0 and args.local_rank in [-1,0]:
                    writer.add_scalar('loss', loss, global_step)
                    writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], global_step)

            if (step+1) % args.log_step == 0 and args.local_rank in [-1,0]:

                elapsed = format_time(time.time() - t0)
                logger.info('Batch {:>5,} of {:>5,}.Loss: {:} Elapsed:{:}.'.format(step+1, len(event_centric_train_dataloader),
                                                                                   format(loss, '.4f'), elapsed))
        avg_loss = np.array(avg_loss).mean()
        return avg_loss,global_step

    elif args.ablation_event or args.ablation_timing:
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = [t.long() for t in batch]
            batch = tuple(t.to(args.device) for t in batch)
            if not macro_loss:
                labels = batch[-1]
            else:
                labels = batch[-2]
            loss, loss_mask = model(*tuple(batch))
            valid_loss_count = cnt_loss(labels, pad_token_id)

            loss = loss.sum() / sum(valid_loss_count)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                pass
            else:

                loss.backward()

            loss = loss.item()
            avg_loss.append(loss)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 * args.gradient_accumulation_steps)
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()
                global_step += 1

            if global_step % 500 == 0 and args.local_rank in [-1, 0]:
                writer.add_scalar('loss', loss, global_step)
                writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], global_step)
            if (step + 1) % args.log_step == 0 and args.local_rank in [-1, 0]:
                elapsed = format_time(time.time() - t0)
                logger.info('Batch {:>5,} of {:>5,}.Loss: {:} Elapsed:{:}.'.format(step + 1,
                                                                                   len(train_dataloader),
                                                                                   format(loss, '.4f'), elapsed))

        avg_loss = np.array(avg_loss).mean()
        return avg_loss, global_step

def evaluate(test_dataloader,model,args):
    perplexity = []
    model.eval()   
    with torch.no_grad():
        for batch in tqdm(test_dataloader,total=len(test_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            batch = [t.long() for t in batch]
            output = model(*tuple(batch))
            loss = output[0]
            perplexity.append(torch.exp(loss).item())
    return perplexity

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   #  set your own configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',default='', help='config file path')
    parser.add_argument('--local_rank', default=-1, type=int)
    config_file = parser.parse_args().config_file
    args = Args(config_file)
    args.local_rank = parser.parse_args().local_rank
    start_date = date.today().strftime('%m-%d')
    if args.local_rank in [-1, 0]:
        log_path = './log/{}/all_{}_500_reverse_{}_{}_{}_{}_{}_{}.log'.format(start_date,args.seed, args.stage1_event_mask_ratio,
                                                  args.stage2_event_mask_ratio, args.loss_ratio_start, args.loss_ratio_end,
                                                                       args.macro_loss, args.annotation)
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
    torch.cuda.empty_cache()
    seed_everything(args.seed)
    if args.multi_gpu and args.use_gpu:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", args.local_rank)
    else :
        args.local_rank = -1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_gpu == False:
        device = torch.device('cpu')
    args.device = device
    logger = None
    if args.local_rank in [-1, 0]:
        logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s | %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            filename=log_path,
            filemode=args.filemode)
        logger = logging.getLogger()
        logger.info("Process rank: {}, device: {}, distributed training: {}".format(
                    args.local_rank,device, bool(args.local_rank != -1)))
        logger.info("Training/evaluation parameters %s", args.to_str())
    model = MODEL_CLASSES[args.model_type](args)
    tokenizer = model.tokenizer
    num_added_toks = tokenizer.add_tokens(['<eoe>', '<before>', '<after>', '<vague>'])
    model.mlm.resize_token_embeddings(len(tokenizer))
    if args.local_rank in [-1, 0]:
        logger.info("We have added %s tokens" % num_added_toks)
    if args.resume:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(dict([(n, p) for n, p in checkpoint['model'].items()]), strict=False)
    if args.noise_lambda != 0:
        for name ,para in model.named_parameters():
            model.state_dict()[name][:] += (torch.rand(para.size())-0.5)*args.noise_lambda*torch.std(para)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    start_epoch = 0
    optimizer = AdamW(optimizer_grouped_parameters,eps = args.epsilon,betas=(0.9,0.999),lr=args.lr)

    if args.ablation_timing and not args.ablation_event:
        train_num = 200000
    elif args.ablation_event and not args.ablation_timing:
        train_num = 200000
    elif args.ablation_event and args.ablation_timing:
        train_num = 100000
    else:
        train_num = 400000
    if args.ablation_timing or args.ablation_event:
        args.num_update_steps_per_epoch = math.ceil(((train_num / (args.gpu_num * args.per_gpu_train_batch_size)))
                                                    / args.gradient_accumulation_steps)
    else:
        args.num_update_steps_per_epoch = math.ceil(((train_num/(args.gpu_num*args.per_gpu_train_batch_size * 2)))
                                                    / args.gradient_accumulation_steps)
    args.total_steps = args.num_train_epochs * args.num_update_steps_per_epoch

    if args.local_rank in [-1, 0]:
        logger.info('overall train epochs: {} | num_update_steps_per_epoch: {}'.format(args.num_train_epochs,
                                                                                       args.num_update_steps_per_epoch))
    lr_schedule_num_training_steps = 50 * args.num_update_steps_per_epoch
    args.num_warmup_steps = 500
    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=lr_schedule_num_training_steps,
        lr_end=1e-7,
        power=3,
    )
    model.to(args.device)
    model.train()
    if args.fp16:
        pass
    if args.local_rank in [-1,0]:
        tensorboard_path = './tensorboard/{}/{}'.format(start_date,args.annotation)
        if not os.path.exists(os.path.dirname(tensorboard_path)):
            os.makedirs(os.path.dirname(tensorboard_path))
        writer = SummaryWriter(tensorboard_path)
    else:
        writer = None
    global_step = 0
    best_performance = 100000
    best_checkpoint_path = None
    macro_loss = args.macro_loss

    if not macro_loss:
        if not args.ablation_event and args.ablation_timing:
            event_centric_data_path = os.path.join(args.data_dir, 'process_' + str(args.stage1_event_mask_ratio) + '_event_centric.pkl')
        elif not args.ablation_timing and args.ablation_event:
            event_temporal_mask_data_path = os.path.join(args.data_dir, 'process_' + str(args.stage2_event_mask_ratio) + '_all_event_temporal_mask.pkl')
        elif args.ablation_event and args.ablation_timing:
            autoregressive_event_centric_data_path = os.path.join(args.data_dir, 'process_' + 'autoregressive_event_centric.pkl')
        else:
            event_centric_data_path = os.path.join(args.data_dir, 'process_' + str(args.stage1_event_mask_ratio) + '_event_centric.pkl')
            event_temporal_mask_data_path = os.path.join(args.data_dir, 'process_' + str(args.stage2_event_mask_ratio) + '_all_event_temporal_mask.pkl')
    else:
        event_centric_data_path = os.path.join(args.data_dir, 'process_' + str(args.stage1_event_mask_ratio) + '_macro_event_centric.pkl')
        event_temporal_mask_data_path = os.path.join(args.data_dir, 'process_' + str(args.stage2_event_mask_ratio) + '_new_macro_event_temporal_mask.pkl')
    train_raw_data_path = os.path.join(args.data_dir, 'train_eot_pretraining_data.json')
    dev_raw_data_path = os.path.join(args.data_dir, 'dev_eot_pretraining_data.json')
    patience = args.patience
    fail_time = 0
    event_centric_train_dataset = None
    event_temporal_mask_train_dataset = None
    autoregressive_event_centric_dataset = None
    dev_dataset = None

    if not event_centric_train_dataset:
        if not macro_loss:
            if not args.ablation_event:
                if os.path.isfile(event_centric_data_path):
                    event_centric_train_dataset = pickle.load(open(event_centric_data_path, 'rb'))
                else:
                    time_start_process = time.time()
                    event_centric_train_raw_dataset = convert_raw_data_to_raw_dataset(train_raw_data_path, args, logger,
                                                                                      mode='event_centric')
                    encode_input_ids, encode_attention_masks, decode_attention_masks, labels = \
                        convert_dataset_to_tensordataset(event_centric_train_raw_dataset, model.tokenizer)

                    time_end_process = format_time(time.time() - time_start_process)
                    event_centric_train_dataset = TensorDataset(encode_input_ids, encode_attention_masks,
                                                                decode_attention_masks, labels)
                    with open(event_centric_data_path, 'wb') as file:
                        pickle.dump(event_centric_train_dataset, file)
        else:
            if os.path.isfile(event_centric_data_path):
                event_centric_train_dataset = pickle.load(open(event_centric_data_path, 'rb'))
            else:
                time_start_process = time.time()
                macro_event_centric_train_raw_dataset = convert_raw_data_to_macro_raw_dataset(train_raw_data_path, args,
                                                                                              logger,
                                                                                              mode='event_centric')
                encode_input_ids, encode_attention_masks, decode_attention_masks, labels, loss_mask = \
                    convert_dataset_to_macro_tensordataset(macro_event_centric_train_raw_dataset, model.tokenizer,
                                                           temporal_mask=False)

                time_end_process = format_time(time.time() - time_start_process)
                event_centric_train_dataset = TensorDataset(encode_input_ids, encode_attention_masks,
                                                            decode_attention_masks, labels, loss_mask)
                with open(event_centric_data_path, 'wb') as file:
                    pickle.dump(event_centric_train_dataset, file)

    if not event_temporal_mask_train_dataset:
        if not macro_loss:
            if not args.ablation_timing:
                if os.path.isfile(event_temporal_mask_data_path):
                    event_temporal_mask_train_dataset = pickle.load(open(event_temporal_mask_data_path, 'rb'))
                else:
                    time_start_process = time.time()
                    event_temporal_mask_train_raw_dataset = convert_raw_data_to_raw_dataset(train_raw_data_path, args,
                                                                                            logger,
                                                                                            mode='event_temporal_mask')
                    encode_input_ids, encode_attention_masks, decode_attention_masks, labels = \
                        convert_dataset_to_tensordataset(event_temporal_mask_train_raw_dataset, model.tokenizer)
                    time_end_process = format_time(time.time() - time_start_process)
                    event_temporal_mask_train_dataset = TensorDataset(encode_input_ids, encode_attention_masks,
                                                                      decode_attention_masks, labels)
                    with open(event_temporal_mask_data_path, 'wb') as file:
                        pickle.dump(event_temporal_mask_train_dataset, file)

        else:
            if os.path.isfile(event_temporal_mask_data_path):
                event_temporal_mask_train_dataset = pickle.load(open(event_temporal_mask_data_path, 'rb'))
            else:
                time_start_process = time.time()
                event_temporal_mask_train_raw_dataset = convert_raw_data_to_macro_raw_dataset(train_raw_data_path, args,
                                                                                              logger,
                                                                                              mode='event_temporal_mask')
                encode_input_ids, encode_attention_masks, decode_attention_masks, labels, loss_mask = \
                    convert_dataset_to_macro_tensordataset(event_temporal_mask_train_raw_dataset, model.tokenizer,
                                                           temporal_mask=True)
                time_end_process = format_time(time.time() - time_start_process)
                event_temporal_mask_train_dataset = TensorDataset(encode_input_ids, encode_attention_masks,
                                                                  decode_attention_masks, labels, loss_mask)
                with open(event_temporal_mask_data_path, 'wb') as file:
                    pickle.dump(event_temporal_mask_train_dataset, file)

    if not autoregressive_event_centric_dataset:
        if not macro_loss:
            if args.ablation_timing and args.ablation_event:
                if os.path.isfile(autoregressive_event_centric_data_path):
                    autoregressive_event_centric_train_dataset = pickle.load(open(autoregressive_event_centric_data_path, 'rb'))
                else:
                    time_start_process = time.time()
                    autoregressive_event_centric_train_raw_dataset = convert_raw_data_to_raw_dataset(train_raw_data_path, args,
                                                                                            logger,
                                                                                            mode='autoregressive_mask')
                    encode_input_ids, encode_attention_masks, decode_attention_masks, labels = \
                        convert_dataset_to_tensordataset(autoregressive_event_centric_train_raw_dataset, model.tokenizer)
                    time_end_process = format_time(time.time() - time_start_process)
                    autoregressive_event_centric_train_dataset = TensorDataset(encode_input_ids, encode_attention_masks,
                                                                      decode_attention_masks, labels)
                    with open(autoregressive_event_centric_data_path, 'wb') as file:
                        pickle.dump(autoregressive_event_centric_train_dataset, file)

        else:
            if os.path.isfile(event_temporal_mask_data_path):
                autoregressive_event_centric_train_dataset = pickle.load(open(autoregressive_event_centric_data_path, 'rb'))
            else:
                time_start_process = time.time()
                autoregressive_event_centric_train_raw_dataset = convert_raw_data_to_macro_raw_dataset(train_raw_data_path, args,
                                                                                              logger,
                                                                                              mode='event_temporal_mask')
                encode_input_ids, encode_attention_masks, decode_attention_masks, labels, loss_mask = \
                    convert_dataset_to_macro_tensordataset(autoregressive_event_centric_train_raw_dataset, model.tokenizer,
                                                           temporal_mask=True)
                time_end_process = format_time(time.time() - time_start_process)
                autoregressive_event_centric_train_dataset = TensorDataset(encode_input_ids, encode_attention_masks,
                                                                  decode_attention_masks, labels, loss_mask)
                with open(autoregressive_event_centric_data_path, 'wb') as file:
                    pickle.dump(autoregressive_event_centric_train_dataset, file)

    if not args.ablation_event:
        event_centric_train_sampler = RandomSampler(event_centric_train_dataset) \
            if args.local_rank == -1 else DistributedSampler(event_centric_train_dataset)
    if not args.ablation_timing:
        event_temporal_mask_train_sampler = RandomSampler(event_temporal_mask_train_dataset) \
            if args.local_rank == -1 else DistributedSampler(event_temporal_mask_train_dataset)

    if args.ablation_event and args.ablation_timing:
        autoregressive_event_centric_train_sampler = RandomSampler(autoregressive_event_centric_train_dataset) \
            if args.local_rank == -1 else DistributedSampler(autoregressive_event_centric_train_dataset)


    if not args.ablation_event:
        event_centric_train_dataloader = DataLoader(event_centric_train_dataset,
                                                    sampler=event_centric_train_sampler,
                                                    batch_size=args.per_gpu_train_batch_size,
                                                    num_workers=0)

    if not args.ablation_timing:
        event_temporal_mask_train_dataloader = DataLoader(event_temporal_mask_train_dataset,
                                                          sampler=event_temporal_mask_train_sampler,
                                                          batch_size=args.per_gpu_train_batch_size,
                                                          num_workers=0)

    if args.ablation_event and args.ablation_timing:
        autoregressive_event_centric_train_dataloader = DataLoader(autoregressive_event_centric_train_dataset,
                                                                   sampler=autoregressive_event_centric_train_sampler,
                                                                   batch_size=args.per_gpu_train_batch_size,
                                                                   num_workers=0)

    if args.local_rank in [-1, 0]:
        if not dev_dataset:
            if os.path.isfile('/workspace/flashback/flashback_gen-main/data/pretrain_data/dev_data.pkl'):
                dev_dataset = pickle.load(
                    open('/workspace/flashback/flashback_gen-main/data/pretrain_data/dev_data.pkl', 'rb'))
            else:
                time_start_process = time.time()
                logger.info('start processing the evaluation dataset!')
                dev_raw_dataset = convert_raw_data_to_raw_dataset(dev_raw_data_path, args, logger, mode='eval')
                dev_raw_data_path = os.path.join(args.data_dir, 'dev_eot_pretraining_data.json')
                encode_input_ids, encode_attention_masks, decode_attention_masks, labels = \
                    convert_dataset_to_tensordataset(dev_raw_dataset, model.tokenizer)
                time_end_process = format_time(time.time() - time_start_process)
                logger.info('The time for processing the evaluation dataset is {:}'.format(time_end_process))
                dev_dataset = TensorDataset(encode_input_ids, encode_attention_masks, decode_attention_masks,
                                            labels)
                with open('/workspace/flashback/flashback_gen-main/data/pretrain_data/dev_data.pkl', 'wb') as file:
                    pickle.dump(dev_dataset, file)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.eval_batch_size, num_workers=0)

    torch.cuda.empty_cache()
    if not args.ablation_event and args.ablation_timing:
        train_dataloader = [event_centric_train_dataloader]
    elif not args.ablation_timing and args.ablation_event:
        train_dataloader = [event_temporal_mask_train_dataloader]
    elif args.ablation_event and args.ablation_timing:
        train_dataloader = [autoregressive_event_centric_train_dataloader]
    else:
        train_dataloader = (event_centric_train_dataloader, event_temporal_mask_train_dataloader)

    for epoch in range(int(args.num_train_epochs)):
        if len(train_dataloader) == 1 and args.local_rank != -1:
            train_dataloader = train_dataloader[0]
            train_dataloader.sampler.set_epoch(epoch)
        elif len(train_dataloader) != 1 and args.local_rank != -1:
            train_dataloader[0].sampler.set_epoch(epoch)
            train_dataloader[1].sampler.set_epoch(epoch)
        elif len(train_dataloader) == 1 and args.local_rank == -1:
            train_dataloader = train_dataloader[0]
        else:
            train_dataloader = train_dataloader

        if fail_time>=patience:
            break
        if epoch < start_epoch:
            continue
        if args.local_rank in [-1,0]:
            logger.info('local_rank={},epoch={}'.format(args.local_rank, epoch))

        train_loss, global_step = train(args, train_dataloader,model,optimizer,lr_scheduler,writer,logger,global_step, macro_loss)
        if args.local_rank in [-1,0]:
            logger.info('epoch={}, loss={}'.format(epoch, train_loss))

        torch.cuda.empty_cache()
        gc.collect()

        if args.local_rank in [-1,0]:

            perplexity = evaluate(dev_dataloader, model, args)
            perplexity = np.mean(perplexity)
            writer.add_scalar('perplexity', perplexity, epoch)

            logger.info("epoch={},perplexity={}".format(epoch,perplexity))

            if perplexity < best_performance:
                checkpoints_path = './checkpoints/{}/{}/all_{}_{}_{}_{}_{}_{}/epoch{}'.format(start_date,args.seed,args.annotation,
                args.stage1_event_mask_ratio,args.stage2_event_mask_ratio, args.loss_ratio_start, args.loss_ratio_end,
                args.macro_loss, epoch)

                if not os.path.exists(checkpoints_path):
                    os.makedirs(checkpoints_path)
                best_checkpoint_path = os.path.join(checkpoints_path,'best_checkpoint.pt')

                model.to(torch.device('cpu'))

                torch.save({'model':model.state_dict(),'epoch':epoch},best_checkpoint_path)
                logger.info('Save best checkpoint to {}'.format(best_checkpoint_path))

                model.to(args.device)

                best_performance = perplexity

                fail_time = 0
                logger.info("best_performance={},best_checkpoint_path={}".format(perplexity,best_checkpoint_path))
            else:
                fail_time+=1

if __name__=='__main__':
    main()
