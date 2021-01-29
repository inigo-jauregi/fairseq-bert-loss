import matplotlib.pyplot as plt
import statistics
import numpy as np
import scipy.stats as stats


nll_entropy = open('inference_entropy_distribution_nll.txt')
list_nll_entropy_vals = []
for line in nll_entropy:
    list_nll_entropy_vals.append(float(line.strip()))

mu_nll = sum(list_nll_entropy_vals) / len(list_nll_entropy_vals)
std_nll = statistics.stdev(list_nll_entropy_vals)
x = np.linspace(0, 7, 1000)
plt.plot(x, stats.norm.pdf(x, mu_nll, std_nll), label='NLL')


raw_entropy = open('inference_entropy_distribution_raw.txt')
list_raw_entropy_vals = []
for line in raw_entropy:
    list_raw_entropy_vals.append(float(line.strip()))

mu_raw = sum(list_raw_entropy_vals) / len(list_raw_entropy_vals)
std_raw = statistics.stdev(list_raw_entropy_vals)
plt.plot(x, stats.norm.pdf(x, mu_raw, std_raw), label='F_BERT (Dense vectors)')


sparsemax_entropy = open('inference_entropy_distribution_sparsemax.txt')
list_sparsemax_entropy_vals = []
for line in sparsemax_entropy:
    list_sparsemax_entropy_vals.append(float(line.strip()))

mu_sparsemax = sum(list_sparsemax_entropy_vals) / len(list_sparsemax_entropy_vals)
std_sparsemax = statistics.stdev(list_sparsemax_entropy_vals)
plt.plot(x, stats.norm.pdf(x, mu_sparsemax, std_sparsemax), label='F_BERT (Sparsemax)')


gumbel_entropy = open('inference_entropy_distribution_gumbel.txt')
list_gumbel_entropy_vals = []
for line in gumbel_entropy:
    list_gumbel_entropy_vals.append(float(line.strip()))

mu_gumbel = sum(list_gumbel_entropy_vals) / len(list_gumbel_entropy_vals)
std_gumbel = statistics.stdev(list_gumbel_entropy_vals)
plt.plot(x, stats.norm.pdf(x, mu_gumbel, std_gumbel), label='F_BERT (Gumbel-Softmax)')


plt.xlabel('Entropy')
plt.ylabel('Density')
#plt.hist(list_nll_entropy_vals, bins=1000, density=True, label='NLL')
plt.plot()
#plt.plot(x_axis, list_bert_prob_entropy, 'y', label='BERT (prob)')
#plt.plot(x_axis, list_bert_1_entropy, 'r', label='BERT (tau=1)')
#plt.plot(x_axis, list_bert_0_1_entropy, 'g', label='BERT (tau=0.1)')
#plt.plot(x_axis, list_bert_0_00001_entropy, 'k', label='BERT (tau=1e-5)')
plt.legend(loc="upper right")
plt.show()
