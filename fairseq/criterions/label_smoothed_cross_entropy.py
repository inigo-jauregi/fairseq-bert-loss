# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from torch.distributions import Categorical

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

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


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing

        # File
        self.loss_stats_file = open('stats_nll_w_fbert.txt', 'w')
        self.loss_stats_file.write('average_entropy\taccuracy\tNLL_loss\tF_BERT_eval\n')

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        print(sample['net_input'])
        net_output = model(**sample['net_input'])
        loss, nll_loss, n_correct, total_n = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'n_correct': n_correct,
            'total_n': total_n
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        my_target = model.get_targets(sample, net_output)

        # Calculate entropy
        # probs = model.get_normalized_probs(net_output, log_probs=False)
        # average_entropy = 0.
        # rows, cols = my_target.size()
        # refs_list = []
        # preds_list = []
        # for i in range(rows):
        #     ref_sentence = []
        #     pred_sentence = []
        #     for j in range(cols):
        #         ref_word = model.decoder.dictionary.__getitem__(my_target[i, j].cpu().detach().numpy())
        #         pred_word = model.decoder.dictionary.__getitem__(probs[i, j].argmax().cpu().detach().numpy())
        #         prob_entropy = Categorical(probs[i, j, :]).entropy().cpu().detach().numpy()
        #         if my_target[i, j] != model.decoder.dictionary.pad_index:
        #             average_entropy += prob_entropy
        #             ref_sentence.append(ref_word)
        #             pred_sentence.append(pred_word)
        #     refs_list.append(" ".join(ref_sentence))
        #     preds_list.append(" ".join(pred_sentence))
        #     print(" ".join(ref_sentence), len(ref_sentence))
        #     # print(" ".join(pred_sentence), len(pred_sentence))
        # average_entropy = average_entropy / (rows * cols)


        # Calculate F-BERT
        # results = score(preds_list, refs_list, model_type='bert-base-uncased', device='cuda:0', verbose=False)
        # f1_avg_results = np.average(results[2].detach().cpu().numpy())
        # print(f1_avg_results)

        # Calculate accuracy
        acc_target = target.squeeze()
        pred = lprobs.max(1)[1].squeeze()
        non_padding = acc_target.ne(model.decoder.dictionary.pad_index).squeeze()
        total_num = non_padding.sum()
        num_correct = pred.eq(acc_target) \
            .masked_select(non_padding) \
            .sum()
        # print(num_correct)
        # print(total_num)
        # print(num_correct.detach().cpu().numpy()/total_num.detach().cpu().numpy())

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        # batch_size = target.size()[0]
        print_acc = (num_correct.detach().cpu().numpy() / total_num.detach().cpu().numpy())*100
        print_nll_loss = nll_loss.detach().cpu().numpy() / total_num.detach().cpu().numpy()

        # self.loss_stats_file.write(str(average_entropy) + '\t' + str(print_acc) + '\t' + str(print_nll_loss) + '\t'
        #                            + str(f1_avg_results) + '\n')

        return loss, nll_loss, num_correct, total_num

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        n_correct = sum(log.get('n_correct', 0) for log in logging_outputs)
        total_n = sum(log.get('total_n', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('accuracy', float(n_correct) / float(total_n), total_n, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
