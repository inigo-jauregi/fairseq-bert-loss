import matplotlib.pyplot as plt

nll_entropy = open('../stats_nll.txt')
list_nll_entropy = []
for line in nll_entropy:
    list_nll_entropy.append(float(line.strip()))

bert_prob_entropy = open('../stats_bert_nll.txt')
list_bert_prob_entropy = []
for line in bert_prob_entropy:
    list_bert_prob_entropy.append(float(line.strip()))

bert_1_entropy = open('../stats_tau_1.txt')
list_bert_1_entropy = []
for line in bert_1_entropy:
    list_bert_1_entropy.append(float(line.strip()))

bert_0_1_entropy = open('../stats_tau_0.1.txt')
list_bert_0_1_entropy = []
for line in bert_0_1_entropy:
    list_bert_0_1_entropy.append(float(line.strip()))

bert_0_00001_entropy = open('../stats_tau_0.00001.txt')
list_bert_0_00001_entropy = []
for line in bert_0_00001_entropy:
    list_bert_0_00001_entropy.append(float(line.strip()))

x_axis = list(range(len(list_nll_entropy)))

plt.plot(x_axis, list_nll_entropy, 'b', label='NLL')
plt.plot(x_axis, list_bert_prob_entropy, 'y', label='BERT (prob)')
plt.plot(x_axis, list_bert_1_entropy, 'r', label='BERT (tau=1)')
plt.plot(x_axis, list_bert_0_1_entropy, 'g', label='BERT (tau=0.1)')
plt.plot(x_axis, list_bert_0_00001_entropy, 'k', label='BERT (tau=1e-5)')
plt.legend(loc="upper right")
plt.show()