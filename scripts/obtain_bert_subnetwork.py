import torch
from fairseq.bert_score.scorer import BERTScorer

my_scorer = BERTScorer('bert-base-uncased')

for name, param in my_scorer._model.named_parameters():
    if name.startswith('embeddings'):
        print(name)
        print(param.size())
        print(type(param))

input_ids = [0]*509 + [1000, 3000, 2395]
out = my_scorer._model(torch.tensor([input_ids]))
print(out[0].size(), out[1].size())
input_emb_one = [-0.2, 0.13, 0.432] * 256
input_emb_zeros = [0.]*768
input_embs = [input_emb_zeros] * 509 + [input_emb_one]*3
out_emb = my_scorer._model(inputs_embeds=torch.tensor([input_embs]))
print(out_emb[0].size(), out_emb[1].size())
