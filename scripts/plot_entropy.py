import matplotlib.pyplot as plt

nll_entropy = open('stats_nll_w_fbert_de_en.txt')
list_nll_entropy = []
list_x_nll_entropy = []
counter = 0
prob_list = []
for line in nll_entropy:
    if counter > 0:
        prob_list.append(float(line.strip().split('\t')[0]))
        if counter % 500 == 0 and counter > 0:
            list_x_nll_entropy.append(counter)
            list_nll_entropy.append(sum(prob_list) / len(prob_list))
            prob_list = []
    counter += 1


raw_entropy = open('stats_aligned_bert_raw.txt')
list_raw_entropy = []
list_x_raw_entropy = []
counter_raw = 0
prob_list = []
for line in raw_entropy:
    if counter_raw > 0:
        prob_list.append(float(line.strip().split('\t')[0]))
        if counter_raw % 500 == 0 and counter_raw > 0:
            list_x_raw_entropy.append(counter_raw + counter - 500)
            list_raw_entropy.append(sum(prob_list) / len(prob_list))
            prob_list = []
    counter_raw += 1

#bert_prob_entropy = open('../stats_bert_nll.txt')
#list_bert_prob_entropy = []
#for line in bert_prob_entropy:
#    list_bert_prob_entropy.append(float(line.strip()))

#bert_1_entropy = open('../stats_tau_1.txt')
#list_bert_1_entropy = []
#for line in bert_1_entropy:
#    list_bert_1_entropy.append(float(line.strip()))

#bert_0_1_entropy = open('../stats_tau_0.1.txt')
#list_bert_0_1_entropy = []
#for line in bert_0_1_entropy:
#    list_bert_0_1_entropy.append(float(line.strip()))

#bert_0_00001_entropy = open('../stats_tau_0.00001.txt')
#list_bert_0_00001_entropy = []
#for line in bert_0_00001_entropy:
#    list_bert_0_00001_entropy.append(float(line.strip()))

# x_axis = list(range(len(list_nll_entropy)))

plt.plot(list_x_nll_entropy, list_nll_entropy, label='NLL')
plt.plot(list_x_raw_entropy, list_raw_entropy, label='F-BERT (Softmax)')
#plt.plot(x_axis, list_bert_prob_entropy, 'y', label='BERT (prob)')
#plt.plot(x_axis, list_bert¸¸¸¸¸¸¸¸¸¸_1_entropy, 'r', label='BERT (tau=1)')
#plt.plot(x_axis, list_bert_0_1_entropy, 'g', label='BERT (tau=0.1)')
#plt.plot(x_axis, list_bert_0_00001_entropy, 'k', label='BERT (tau=1e-5)')
plt.legend(loc="upper right")
plt.show()
