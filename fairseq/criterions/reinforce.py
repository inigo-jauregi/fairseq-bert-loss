# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch
from fairseq import metrics
from dataclasses import dataclass, field
from fairseq.bert_score import BERTScorer

from collections import defaultdict
from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion
# from fairseq.custom.evaluate_utils import batch_input_sequence_by_prefix_length
# from fairseq.custom.metrics import ngram_metrics
# from fairseq.custom.metrics import TrainingMetrics
# from fairseq.dataclass import FairseqDataclass


# @dataclass
# class SequencePenaltyCriterionConfig(FairseqDataclass):
#     sequence_ngram_n: int = field(
#         default=4,
#         metadata={"help": "number of repeated n-grams wanting to penalise"},
#     )
#     sequence_prefix_length: int = field(
#         default=50,
#         metadata={"help": "length of prefix input?"},
#     )
#     sequence_completion_length: int = field(
#         default=100,
#         metadata={"help": "how long the predicted sequence will be?"},
#     )
#     sequence_candidate_type: str = field(
#         default="repeat",
#         metadata={"help": "candidate type for penalty (repeat, random)"},
#     )
#     mask_p: float = field(
#         default=0.5,
#         metadata={"help": "float between 0 and 1 that identifies random candidates in sequence to penalize?"},
#     )


@register_criterion('reinforce')
class Reinforce(FairseqCriterion):

    def __init__(self, task, bert_model, max_len_decoding):
        super().__init__(task)

        self.bert_model = bert_model
        self.max_len_decoding = max_len_decoding

        self.bert_scorer = BERTScorer(self.bert_model)  # , device='cpu')
        self.pad_token_id = self.bert_scorer._tokenizer.convert_tokens_to_ids('[PAD]')

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--bert-model', default='bert-base-uncased', type=str, metavar='D',
                            help='pretrained BERT model to calculate BERT loss')
        parser.add_argument('--max-len-decoding', default=20, type=int, metavar='D',
                            help='maximum length when decoding a sentence.')

    def forward(self, model, sample, reduce=True, generator=None):

        # Targets for reward computation
        target = sample['target']

        # Forward encoder
        encoder_out = model.encoder(
            sample['net_input']['src_tokens'], src_lengths=sample['net_input']['src_lengths'],
            return_all_hiddens=True
        )

        # Decode translations sequentially (greedy decoding)
        pred_toks, lprobs = sequential_decoding(model, encoder_out, max_len_decoding=self.max_len_decoding)
        # Extract prob values
        pred_toks_col = pred_toks.view(-1, 1).squeeze()
        lprobs_col = lprobs.view(-1, lprobs.size()[-1])
        lprobs_col = lprobs_col[torch.arange(lprobs_col.size()[0]), pred_toks_col]
        lprobs_back = lprobs_col.view(pred_toks.size())
        lprobs_added = lprobs_back.sum(axis=1)
        # print(lprobs_added.size())
        # print(pred_toks.size())

        # Compute F-BERT
        # rewards = score(preds_list, refs_list, model_type='bert-base-uncased', device='cuda:0', verbose=False)
        rewards = self.bert_scorer.bert_loss_calculation(pred_toks, target, pad_token_id=self.pad_token_id,
                                                         both_tensors=True, out_type='f1_batch')
        # Detach rewards from the loss function
        rewards = rewards.detach()
        f_bert = rewards.sum()

        loss = - rewards * lprobs_added
        loss = loss.sum()

        # Calculate accuracy
        acc_target = target.view(-1, 1).squeeze()
        pred = lprobs.contiguous().view(-1, lprobs.size(-1)).max(1)[1]
        non_padding = acc_target.view(-1, 1).ne(model.decoder.dictionary.pad_index).squeeze()
        total_num = non_padding.sum()
        num_correct = pred.eq(acc_target) \
            .masked_select(non_padding) \
            .sum()

        return loss, f_bert, num_correct, total_num

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        f_bert_sum = sum(log.get('f_bert', 0) for log in logging_outputs)
        # print('Sum ', f_bert_sum)
        # ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        n_sentences = sum(log.get('n_sentences', 0) for log in logging_outputs)
        # print('Avg ', f_bert_sum / n_sentences)
        n_correct = sum(log.get('n_correct', 0) for log in logging_outputs)
        total_n = sum(log.get('total_n', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / n_sentences, n_sentences, round=3)
        metrics.log_scalar('f_bert', f_bert_sum / n_sentences, n_sentences, round=3)
        metrics.log_scalar('accuracy', float(n_correct) / float(total_n), total_n, round=3)
        # metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))


# Test
def _forward_one(model, encoded_source, tokens, incremental_states=None, temperature=1., return_attn=False,
                 return_logits=False, **decoder_kwargs):
    if incremental_states is not None:
        decoder_out = list(model.decoder(tokens, encoded_source, incremental_state=incremental_states,
                                         **decoder_kwargs))
    else:
        decoder_out = list(model.decoder(tokens, encoded_source, **decoder_kwargs))
    decoder_out[0] = decoder_out[0][:, -1:, :]
    # print(decoder_out[0].size())
    if temperature != 1.:
        decoder_out[0].div_(temperature)
    attn = decoder_out[1]
    if type(attn) is dict:
        attn = attn['attn'][0]
    attn = None
    if attn is not None:
        if type(attn) is dict:
            attn = attn['attn']
        attn = attn[:, :, -1, :]  # B x L x t
    if return_logits:
        logits_t = decoder_out[0][:, -1, :]
        return logits_t, attn
    log_probs = model.get_normalized_probs(decoder_out, log_probs=True)
    log_probs = log_probs[:, -1, :]
    return log_probs, attn


def sequential_decoding(model, encoded_source, max_len_decoding):
    # model.eval()
    pred_toks = []
    batch_size = encoded_source[0].size()[1]
    eos_token_id = torch.tensor(model.decoder.dictionary.eos())
    pad_token_id = torch.tensor(model.decoder.dictionary.pad())
    context = torch.tensor([model.decoder.dictionary.bos()]*batch_size).unsqueeze(1)
    states = {}
    all_lprobs = []
    masking_matrix = []
    aux_masking_matrix = []

    for tstep in range(max_len_decoding):
        lprobs, attn_t = _forward_one(model, encoded_source, context, incremental_states=states)
        lprobs[:, pad_token_id] = -math.inf  # never select pad  (MAYBE I CAN ADD MIN LENGTH?)
        pred_tok = lprobs.argmax(dim=1, keepdim=True)
        # Check if predicted token is <eos>
        pred_token_bool = torch.where(pred_tok == eos_token_id, torch.tensor(1.0), torch.tensor(0.0))
        if len(aux_masking_matrix) > 0:
            pred_token_bool = torch.logical_or(aux_masking_matrix[-1], pred_token_bool)
            pred_token_bool = torch.where(pred_token_bool == True, torch.tensor(1.0), torch.tensor(0.0))
            see_if_previous_was_eos = torch.logical_or(masking_matrix[-1], aux_masking_matrix[-1])
            pred_token_bool_true = torch.logical_and(see_if_previous_was_eos, pred_token_bool)
            masking_matrix.append(pred_token_bool_true)
        else:
            masking_matrix.append(torch.zeros(pred_token_bool.size()))
        aux_masking_matrix.append(pred_token_bool)

        pred_toks.append(pred_tok)
        context = torch.cat((context, pred_tok), 1)
        all_lprobs.append(lprobs)
        count_token = pred_token_bool[pred_token_bool == 0].size()[0]
        if count_token == 0:
            break

    # for tok in pred_toks:
    #     print(model.decoder.dictionary.__getitem__(tok[0]))
    masking_matrix = torch.cat(masking_matrix, 1)
    pred_toks = torch.cat(pred_toks, 1)
    all_lprobs = torch.stack(all_lprobs, 1)

    # Apply masking (padding tokens after the <eos> token.)
    pred_toks[masking_matrix == 1.0] = pad_token_id
    # Apply masking (set probability values to zero)
    all_lprobs[masking_matrix == 1.0] = torch.zeros(all_lprobs.size()[-1])

    return pred_toks, all_lprobs
