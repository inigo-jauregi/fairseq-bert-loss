import os
import sys

from nltk.translate.bleu_score import sentence_bleu


# def BLEU_score(self, predictions, ground_truth):
#     batch_size = len(predictions)
#     n_best = len(predictions[0])
#     # print (n_best)
#     # print (predictions[0][0])
#     # print (predictions[0][1])
#     bleu_scores = []
#     for i in range(batch_size):
#         bs_nBest = []
#         for j in range(n_best):
#             pred = predictions[i][j]
#             # print (pred)
#             gt = [ground_truth[i]]
#             # print (gt)
#             BLEU = sentence_bleu(gt, pred) * 100
#             bs_nBest.append(BLEU)
#
#         bleu_scores.append(bs_nBest)
#
#     bleu_scores = torch.FloatTensor(bleu_scores).cuda()  # Cuda line
#
#     # print (bleu_scores.size())
#
#     return bleu_scores.squeeze()


def main(preds_path, refs_path, output_path):

    refs_file=open(refs_path)
    hyps_file=open(preds_path)

    list_refs=[]
    for sen in refs_file:
        list_refs.append(sen)
    list_hyps=[]
    for sen in hyps_file:
        list_hyps.append(sen)


    hyps_BLEUS=[]
    for i in range(len(list_refs)):
        #Create tmp_ref.txt, tmp_hyp_BASELINE.txt and tmp_hyp_NLL_COS.txt files
        # write_ref=open('SINGLE_BLEU/tmp_ref.txt','w')
        # write_ref.write(list_refs[i])
        # write_ref.close()
        # write_hyp = open('SINGLE_BLEU/tmp_hyp.txt', 'w')
        # write_hyp.write(list_hyps[i])
        # write_hyp.close()
        # #COMPUTE BLUE SCORES
        # os.system('perl ../mosesdecoder/scripts/generic/multi-bleu.perl '
        #           'SINGLE_BLEU/tmp_ref.txt < SINGLE_BLEU/tmp_hyp.txt > SINGLE_BLEU/tmp_result.txt')
        # bleu_file=open('SINGLE_BLEU/tmp_result.txt')
        # bleu_line=bleu_file.readline()
        # bleu_score=float(bleu_line.split()[2].replace(',',''))
        bleu_score = sentence_bleu([list_refs[i]], list_hyps[i]) * 100
        print(bleu_score)
        hyps_BLEUS.append(bleu_score)


    # Sentence level
    file_summary=open(output_path,'w')
    for j in range(len(hyps_BLEUS)):
        bleu_score=hyps_BLEUS[j]
        file_summary.write(str(bleu_score)+'\n')
    file_summary.close()

    # Corpus average
    file_summary_avg = open(output_path + '.avg', 'w')
    bleu_score_avg = float(sum(hyps_BLEUS)) / len(hyps_BLEUS)
    file_summary_avg.write(str(bleu_score_avg) + '\n')
    file_summary_avg.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])