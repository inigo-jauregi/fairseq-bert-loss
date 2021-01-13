# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch.nn.functional import gumbel_softmax
from torch.nn import CosineEmbeddingLoss
from torch.distributions import Categorical

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.bert_score import BERTScorer
from fairseq.bert_score.utils import custom_masking, custom_bert_encode

from fairseq.bert_score.score import score
import numpy as np


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
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


@register_criterion('mixed_nll_bert_loss')
class MixedNLLBertLossCriterion(FairseqCriterion):

    def __init__(self, task, bert_model, marginalization, tau_gumbel_softmax, hard_gumbel_softmax, eps_gumbel_softmax,
                 label_smoothing, soft_bert_score, mixed_proportion):
        super().__init__(task)

        self.bert_model = bert_model

        self.marginalization = marginalization

        self.bert_scorer = BERTScorer(self.bert_model, soft_bert_score=soft_bert_score)  # , device='cpu')
        self.pad_token_id = self.bert_scorer._tokenizer.convert_tokens_to_ids('[PAD]')

        # Gumbel-Softmax hyperparameters
        self.tau_gumbel_softmax = tau_gumbel_softmax
        self.hard_gumbel_softmax = hard_gumbel_softmax
        self.eps_gumbel_softmax = eps_gumbel_softmax

        # NLL parameters
        self.eps = label_smoothing

        # Cosine loss
        self.cos_loss = CosineEmbeddingLoss(reduction='sum')

        self._lambda = torch.tensor(mixed_proportion).to(self.bert_scorer.device)

        # File
        self.loss_stats_file = open('stats_mixed_nll_bert_sparsemax.txt', 'w')
        self.loss_stats_file.write('accuracy\tF_BERT\tLoss\n')

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
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--soft-bert-score', action="store_true",
                            help='Whether we compute a soft BERT score or a hard bert-score')
        parser.add_argument('--mixed-proportion', default=0.5, type=float, metavar='D',
                            help='Value')
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
        loss, coss_loss, n_correct, total_n = self.compute_loss(model, net_output, sample, reduce=reduce)
        # print(loss)
        sample_size = sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'coss_loss': coss_loss.data,
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
        # gsm_samples = self.sparsemax(lprobs, 2)
        # print(gsm_samples.size())

        lprobs_nll = lprobs.view(-1, lprobs.size(-1))
        target_nll = model.get_targets(sample, net_output).view(-1, 1)

        # torch.manual_seed(1)  # Seed only seems to affect here
        # print(lprobs.size())
        # print(lprobs[0, 0, :].max())
        # print(probs[0, 0, :].max())
        # lprobs = lprobs.view(-1, lprobs.size(-1))

        # print(self.tau_gumbel_softmax)
        # print(self.hard_gumbel_softmax)
        # print(self.eps_gumbel_softmax)
        rewe = None
        if len(net_output) == 3:
            gsm_samples = net_output[2]
            rewe = True
        elif self.marginalization == 'raw':
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
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs_nll, target_nll, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        # f_bert = self.bert_scorer.bert_loss_calculation(gsm_samples, target, pad_token_id=self.pad_token_id, rewe=rewe)
        # cos_loss = - f_bert
        target_contextual_embs, mask = self.target_contextual_embs(target, self.bert_scorer.device)
        pred_contextual_embs = self.pred_contextual_embs(gsm_samples, mask, rewe=rewe)
        target_contextual_embs_v = target_contextual_embs.view(-1, target_contextual_embs.size()[-1])
        pred_contextual_embs_v = pred_contextual_embs.view(-1, target_contextual_embs.size()[-1])
        cos_loss = self.cos_loss(pred_contextual_embs_v, target_contextual_embs_v,
                             torch.tensor(1.0).to(self.bert_scorer.device))
        # print(f_bert / rows)

        # loss = -torch.log(f_bert)
        # print (nll_loss, ' + ', cos_loss, ' = ', loss)
        loss = (torch.tensor(1.0).to(self.bert_scorer.device) - self._lambda)*nll_loss + self._lambda*cos_loss

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

        # loss = -f_bert

        batch_size = target.size()[0]

        # print('Accuracy: ', (num_correct.detach().cpu().numpy() / total_num.detach().cpu().numpy())*100, '%')
        # print('F-Bert: ', (f_bert/batch_size).detach().cpu().numpy())
        print_acc = (num_correct.detach().cpu().numpy() / total_num.detach().cpu().numpy())*100
        print_f1 = (cos_loss/batch_size).detach().cpu().numpy()
        self.loss_stats_file.write(str(print_acc) + '\t' + str(print_f1) + '\t' +
                                   str(loss.detach().cpu().numpy()) + '\n')

        return loss, cos_loss, num_correct, total_num

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        coss_loss_sum = sum(log.get('coss_loss', 0) for log in logging_outputs)
        # print('Sum ', f_bert_sum)
        # ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        n_sentences = sum(log.get('n_sentences', 0) for log in logging_outputs)
        # print('Avg ', f_bert_sum / n_sentences)
        n_correct = sum(log.get('n_correct', 0) for log in logging_outputs)
        total_n = sum(log.get('total_n', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / n_sentences, n_sentences, round=3)
        metrics.log_scalar('cos_loss', coss_loss_sum / total_n, total_n, round=3)
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

    def sparsemax(self, input, dim_selected=-1):
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
        range = torch.arange(start=1, end=number_of_logits + 1, step=1,
                             device=self.bert_scorer.device, dtype=input.dtype).view(1, -1)
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
        output = torch.max(torch.zeros_like(input).to(self.bert_scorer.device), input - taus)

        # Reshape back to original shape
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, dim_selected)

        return output

    def target_contextual_embs(self, target_ids, device, all_layers=False):

        mask = custom_masking(target_ids, self.pad_token_id, device)

        target_bert_embeddings = custom_bert_encode(
            self.bert_scorer._model, target_ids, attention_mask=mask, all_layers=all_layers
        )

        return target_bert_embeddings, mask

    def pred_contextual_embs(self, preds_tensor, mask, all_layers=False, rewe=None):

        batch_size, max_seq_len, vocab_size = preds_tensor.size()
        emb_size = self.bert_scorer._emb_matrix.size()[-1]
        if rewe:
            preds_tensor_embs = preds_tensor
        else:
            preds_tensor_embs = torch.mm(preds_tensor.contiguous().view(-1, vocab_size),
                                         self.bert_scorer._emb_matrix)
            preds_tensor_embs = preds_tensor_embs.view(-1, max_seq_len, emb_size)

        preds_bert_embedding = custom_bert_encode(
            self.bert_scorer._model, preds_tensor_embs, attention_mask=mask,
            all_layers=all_layers, embs=True
        )

        return preds_bert_embedding
