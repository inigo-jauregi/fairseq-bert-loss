# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch.nn.functional import gumbel_softmax
from torch.distributions import Categorical

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.bert_score import BERTScorer

from fairseq.bert_score.score import score
import numpy as np


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

    def __init__(self, task, bert_model, marginalization, tau_gumbel_softmax, hard_gumbel_softmax, eps_gumbel_softmax,
                 soft_bert_score):
        super().__init__(task)

        self.bert_model = bert_model

        self.marginalization = marginalization

        self.bert_scorer = BERTScorer(self.bert_model, soft_bert_score=soft_bert_score)  # , device='cpu')
        self.pad_token_id = self.bert_scorer._tokenizer.convert_tokens_to_ids('[PAD]')

        # Gumbel-Softmax hyperparameters
        self.tau_gumbel_softmax = tau_gumbel_softmax
        self.hard_gumbel_softmax = hard_gumbel_softmax
        self.eps_gumbel_softmax = eps_gumbel_softmax

        # File
        self.loss_stats_file = open('stats_bert_GUMBEL_TAU_1e-5_FT.txt', 'w')
        self.loss_stats_file.write('average_entropy\taccuracy\tF_BERT\tF_BERT_eval\n')

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--bert-model', default='bert-base-uncased', type=str, metavar='D',
                            help='pretrained BERT model to calculate BERT loss')
        parser.add_argument('--marginalization', default='raw', type=str, metavar='D',
                            help='Embedding marginalization method.')
        parser.add_argument('--tau-gumbel-softmax', default=1.0, type=float,
                            help='Hyper-parameter tau in Gumbel-Softmax')
        parser.add_argument('--hard-gumbel-softmax', action="store_true",
                            help='Whether is a soft or hard sample (i.e. one-hot encoding)')
        parser.add_argument('--eps-gumbel-softmax', default=1e-10, type=float,
                            help='Whether is a soft or hard sample (i.e. one-hot encoding)')
        parser.add_argument('--soft-bert-score', action="store_true",
                            help='Whether we compute a soft BERT score or a hard bert-score')
        # parser.add_argument("--bos", default="<s>", type=str,
        #                     help="Specify bos token from the dictionary.")
        # parser.add_argument("--pad", default="<pad>", type=str,
        #                     help="Specify bos token from the dictionary.")
        # parser.add_argument("--eos", default="</s>", type=str,
        #                     help="Specify bos token from the dictionary.")
        # parser.add_argument("--unk", default="<unk>", type=str,
        #                     help="Specify bos token from the dictionary.")
        # parser.add_argument("--tgtdict_add_sentence_limit_words_after", action="store_true",
        #                     help="Add sentence limit words (i.e. bos, eos, pad, unk) after loading tgtdict.")
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
        # print(net_output[0].size())
        # torch.autograd.set_detect_anomaly(True)

        # TODO: Print vocab of bert and encoder
        loss, f_bert, n_correct, total_n = self.compute_loss(model, net_output, sample, reduce=reduce)
        # print(loss)
        sample_size = sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'f_bert': f_bert.data,
            # 'ntokens': sample['ntokens'],
            'n_sentences': sample['target'].size(0),
            'sample_size': sample_size,
            'n_correct': n_correct,
            'total_n': total_n
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # gsm_samples = model.get_normalized_probs(net_output, log_probs=False)
        # print(lprobs.size())
        # gsm_samples = self.sparsemax(lprobs,2)
        # print(gsm_samples.size())
        # torch.manual_seed(1)  # Seed only seems to affect here
        # print(lprobs.size())
        # print(lprobs[0, 0, :].max())
        # print(probs[0, 0, :].max())
        # lprobs = lprobs.view(-1, lprobs.size(-1))

        # print(self.tau_gumbel_softmax)
        # print(self.hard_gumbel_softmax)
        # print(self.eps_gumbel_softmax)
        if self.marginalization == 'raw':
            gsm_samples = model.get_normalized_probs(net_output, log_probs=False)
        elif self.marginalization == 'sparsemax':
            gsm_samples = self.sparsemax(lprobs, 2)
        elif self.marginalization == 'gumbel-softmax':
            gsm_samples = gumbel_softmax(lprobs, tau=self.tau_gumbel_softmax, hard=self.hard_gumbel_softmax,
                                         eps=self.eps_gumbel_softmax, dim=-1)
        # gsm_samples_2 = gumbel_softmax(lprobs, tau=1, hard=False, eps=1e-10, dim=-1)
        # print(gsm_samples.size())
        # print(gsm_samples[0, 0, :].max())
        # print(gsm_samples_2[0, 0, :].max())
        # print(gsm_samples.device)
        target = model.get_targets(sample, net_output)

        # Calculate entropy
        # probs = model.get_normalized_probs(net_output, log_probs=False)
        # average_entropy = 0.
        # rows, cols = target.size()
        # refs_list = []
        # preds_list = []
        # for i in range(rows):
        #     ref_sentence = []
        #     pred_sentence = []
        #     for j in range(cols):
        #         ref_word = model.decoder.dictionary.__getitem__(target[i, j].cpu().detach().numpy())
        #         pred_word = model.decoder.dictionary.__getitem__(gsm_samples[i, j].argmax().cpu().detach().numpy())
        #         prob_entropy = Categorical(gsm_samples[i,j,:]).entropy().cpu().detach().numpy()
        #         if target[i, j] != self.pad_token_id:
        #             average_entropy += prob_entropy
        #             ref_sentence.append(ref_word)
        #             pred_sentence.append(pred_word)
        #     refs_list.append(" ".join(ref_sentence))
        #     preds_list.append(" ".join(pred_sentence))
        #     # print('Tgt:  ', " ".join(tmp_target_words))
        #     # print('Pred:  ', " ".join(tmp_pred_words))
        # average_entropy = average_entropy / (rows*cols)

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
        f_bert = self.bert_scorer.bert_loss_calculation(gsm_samples, target, pad_token_id=self.pad_token_id)
        # print(f_bert / rows)

        # loss = -torch.log(f_bert)

        # Calculate F-BERT
        # results = score(preds_list, refs_list, model_type='bert-base-uncased', device='cuda:0', verbose=False)
        # f1_avg_results = np.average(results[2].detach().cpu().numpy())
        # print(f1_avg_results)

        # Calculate accuracy
        acc_target = target.view(-1, 1).squeeze()
        pred = gsm_samples.contiguous().view(-1, gsm_samples.size(-1)).max(1)[1]
        non_padding = acc_target.view(-1, 1).ne(model.decoder.dictionary.pad_index).squeeze()
        total_num = non_padding.sum()
        num_correct = pred.eq(acc_target) \
            .masked_select(non_padding) \
            .sum()

        loss = -f_bert

        batch_size = target.size()[0]

        # print('Accuracy: ', (num_correct.detach().cpu().numpy() / total_num.detach().cpu().numpy())*100, '%')
        # print('F-Bert: ', (f_bert/batch_size).detach().cpu().numpy())
        # print_acc = (num_correct.detach().cpu().numpy() / total_num.detach().cpu().numpy())*100
        # print_f1 = (f_bert/batch_size).detach().cpu().numpy()
        # self.loss_stats_file.write(str(average_entropy) + '\t' + str(print_acc) + '\t' +
        #                            str(f_bert.detach().cpu().numpy()) + '\t' + str(print_f1) + '\n')

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

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    @staticmethod
    def sparsemax(input, dim_selected=-1):
        """sparsemax.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, dim_selected)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device='cuda:0', dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        output = torch.max(torch.zeros_like(input).to('cuda:0'), input - taus)

        # Reshape back to original shape
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, dim_selected)

        return output
