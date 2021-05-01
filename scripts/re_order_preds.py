import sys
import os
import numpy as np

def main(output_path, hyp_pred_path, hyp_pred_path_ordered):


    output = open(output_path)
    hyp_pred = open(hyp_pred_path)

    hyp_num_list = []
    for sen in output:
        if sen.startswith('H-'):
            hyp_num = int(sen.split('\t')[0].replace('H-', ''))
            hyp_num_list.append(hyp_num)

    hyp_pred_list = []
    for sen in hyp_pred:
        hyp_pred_list.append(sen)

    assert len(hyp_num_list) == len(hyp_pred_list)

    reordered_hyps_list = [y for x, y in sorted(zip(hyp_num_list, hyp_pred_list))]

    write_file = open(hyp_pred_path_ordered, 'w')
    for sen in reordered_hyps_list:
        write_file.write(sen)
    write_file.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
