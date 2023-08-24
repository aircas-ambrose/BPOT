from sre_constants import RANGE
from torch import nn
from transformers.models.bart.modeling_bart import BartForConditionalGeneration,BartLearnedPositionalEmbedding
from transformers.modeling_outputs import Seq2SeqLMOutput
from models.base.cot import ComplementEntropy
from torch.nn import CrossEntropyLoss
import torch
# torch.set_printoptions(precision=8,sci_mode=False)
import random
from transformers import BartTokenizer

def shift_tokens_right(input_ids, pad_token_id):
    prev_input_ids = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_input_ids[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_input_ids[:, 1:] = input_ids[:, :-1]
    return prev_input_ids

class bart_mask_random(nn.Module):
    def __init__(self, args):
        super(bart_mask_random, self).__init__()
        self.mlm = BartForConditionalGeneration.from_pretrained(args.pretrained_model_path)
        self.tokenizer = BartTokenizer.from_pretrained(args.pretrained_model_path)
        self.args = args
        self.config = self.mlm.config
        self.ce_loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    def forward(
        self,encode_inputs,encode_masks,decode_masks,labels,loss_mask=None
    ):
        if self.args.pretrain and self.training:
            max_encode_len = torch.max(encode_inputs.ne(self.tokenizer.pad_token_id).sum(dim=1))
            max_decode_len = torch.max(labels.ne(self.tokenizer.pad_token_id).sum(dim=1))
            encode_inputs = encode_inputs[:, :max_encode_len].contiguous()
            # decode_inputs = decode_inputs[:, :max_decode_len]
            encode_masks = encode_masks[:, :max_encode_len].contiguous()
            decode_masks = decode_masks[:, :max_decode_len].contiguous()
            labels = labels[:, :max_decode_len]
            labels[labels == self.config.pad_token_id] = -100
            labels = labels.contiguous()
            if loss_mask is not None:
                loss_mask = loss_mask[:, :max_decode_len]
                loss_mask = loss_mask.contiguous()
            output = self.mlm(
                input_ids=encode_inputs,
                attention_mask=encode_masks,
                labels=labels,
                decoder_attention_mask = decode_masks)
            loss = self.ce_loss_fct(output[1].view(-1, output[1].shape[-1]), labels.view(-1))
            loss = loss.view(labels.size(0), -1)
            if loss_mask is not None:
                return loss, loss_mask
            else:
                return loss, None
        else:
            max_encode_len = torch.max(encode_inputs.ne(self.tokenizer.pad_token_id).sum(dim=1) - 1) + 1
            max_decode_len = torch.max(labels.ne(self.tokenizer.pad_token_id).sum(dim=1) - 1) + 1
            encode_inputs = encode_inputs[:, :max_encode_len].contiguous()
            encode_masks = encode_masks[:, :max_encode_len].contiguous()
            decode_masks = decode_masks[:, :max_decode_len].contiguous()
            labels = labels[:, :max_decode_len]
            labels[labels == self.config.pad_token_id] = -100
            labels = labels.contiguous()
            output = self.mlm(
                input_ids=encode_inputs,
                attention_mask=encode_masks,
                labels=labels,
                decoder_attention_mask=decode_masks)

            return output
