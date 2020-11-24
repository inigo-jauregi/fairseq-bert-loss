# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch.nn.functional import gumbel_softmax

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.bert_score import BERTScorer


def bert_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('bert_loss')
class BertLossCriterion(FairseqCriterion):

    def __init__(self, task, bert_model, tau_gumbel_softmax, hard_gumbel_softmax, eps_gumbel_softmax):
        super().__init__(task)

        self.bert_model = bert_model

        self.bert_scorer = BERTScorer(self.bert_model)
        self.pad_token_id = self.bert_scorer._tokenizer.convert_tokens_to_ids('[PAD]')

        # Gumbel-Softmax hyperparameters
        self.tau_gumbel_softmax = tau_gumbel_softmax
        self.hard_gumbel_softmax = hard_gumbel_softmax
        self.eps_gumbel_softmax = eps_gumbel_softmax

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--bert-model', default='bert-base-uncased', type=str, metavar='D',
                            help='pretrained BERT model to calculate BERT loss')
        parser.add_argument('--tau-gumbel-softmax', default=1.0, type=float,
                            help='Hyper-parameter tau in Gumbel-Softmax')
        parser.add_argument('--hard-gumbel-softmax', default=False, type=bool,
                            help='Whether is a soft or hard sample (i.e. one-hot encoding)')
        parser.add_argument('--eps-gumbel-softmax', default=1e-10, type=float,
                            help='Whether is a soft or hard sample (i.e. one-hot encoding)')
        parser.add_argument("--bos", default="<s>", type=str,
                            help="Specify bos token from the dictionary.")
        parser.add_argument("--pad", default="<pad>", type=str,
                            help="Specify bos token from the dictionary.")
        parser.add_argument("--eos", default="</s>", type=str,
                            help="Specify bos token from the dictionary.")
        parser.add_argument("--unk", default="<unk>", type=str,
                            help="Specify bos token from the dictionary.")
        parser.add_argument("--tgtdict_add_sentence_limit_words_after", action="store_true",
                            help="Add sentence limit words (i.e. bos, eos, pad, unk) after loading tgtdict.")
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # net_output: tuple (torch.tensor logits, dict(attn, inner states)
        net_output = model(**sample['net_input'])
        print(net_output[0].size())

        # TODO: Print vocab of bert and encoder
        loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['ntokens']
        logging_output = {
            'f1_loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # probs = model.get_normalized_probs(net_output, log_probs=False)
        # torch.manual_seed(1)  # Seed only seems to affect here
        # print(lprobs.size())
        # print(lprobs[0, 0, :].max())
        # print(probs[0, 0, :].max())
        # lprobs = lprobs.view(-1, lprobs.size(-1))

        # print(self.tau_gumbel_softmax)
        # print(self.hard_gumbel_softmax)
        # print(self.eps_gumbel_softmax)
        gsm_samples = gumbel_softmax(lprobs, tau=self.tau_gumbel_softmax, hard=self.hard_gumbel_softmax,
                                     eps=self.eps_gumbel_softmax, dim=-1)
        # gsm_samples_2 = gumbel_softmax(lprobs, tau=1, hard=False, eps=1e-10, dim=-1)
        # print(gsm_samples.size())
        # print(gsm_samples[0, 0, :].max())
        # print(gsm_samples_2[0, 0, :].max())

        target = model.get_targets(sample, net_output)
        # print(target[0, 10])
        # print(len(model.decoder.dictionary.symbols))
        # print(self.bert_scorer._tokenizer.vocab_size)
        # print(model.decoder.dictionary.__getitem__(target[0, 10]))
        # print(self.bert_scorer._tokenizer.convert_ids_to_tokens(int(target[0, 10])))
        # # print(target)
        # print(target.size())
        # loss, nll_loss = label_smoothed_nll_loss(
        #     lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        # )
        loss = self.bert_scorer.bert_loss_calculation(gsm_samples, target, pad_token_id=self.pad_token_id)
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
