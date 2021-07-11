import matplotlib.pyplot as plt

nll_entropy = open('../stats_aligned_bert_raw.txt')
list_nll_entropy = []
list_nll_accuracy = []
list_nll_nll = []
list_nll_f1_eval = []
headline = False
for line in nll_entropy:
    if headline:
        list_nll_accuracy.append(float(line.strip().split('\t')[0]))
        # list_nll_nll.append(float(line.strip().split('\t')[2]))
        list_nll_f1_eval.append(float(line.strip().split('\t')[1]))
        # splitted_score = line.strip().split('\t')[2].split('.')
        # list_nll_f1_eval.append(float(".".join([splitted_score[1][-2:], splitted_score[2]])))
    else:
        headline = True

x_axis = list(range(len(list_nll_entropy)))

x_axis_avg = []
average_y = []
batch_size = 967
for i in range(len(list_nll_f1_eval) // batch_size):
    average_y.append(sum(list_nll_f1_eval[i*batch_size:i*batch_size+batch_size]) / batch_size)
    x_axis_avg.append(i*batch_size+batch_size)


# print(len(list_nll_f1_eval) // batch_size)


plt.plot(x_axis_avg, average_y, 'b', label='F1-Eval')
plt.ylabel('Accuracy (%)')
# plt.ylabel('F-BERT (%)')
plt.title('BERT_loss Fine-Tuning (Dense-Vectors)')
plt.show()