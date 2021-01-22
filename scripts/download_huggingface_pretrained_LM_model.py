import os
from transformers import AutoConfig, AutoTokenizer, AutoModel

bert_model_name = "DeepPavlov/rubert-base-cased"
if os.path.isdir('../pretrained-LMs/'+bert_model_name):
    os.mkdir('../pretrained-LMs/'+bert_model_name)

tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
model_config = AutoConfig.from_pretrained(bert_model_name)
model = AutoModel.from_pretrained(bert_model_name)
tokenizer.save_pretrained('../pretrained-LMs/'+bert_model_name)
model.save_pretrained('../pretrained-LMs/'+bert_model_name)
model_config.save_pretrained('../pretrained-LMs/'+bert_model_name)

vocab_file = open('../pretrained-LMs/'+bert_model_name+'/vocab.txt')
list_lines = []
for line in vocab_file:
    list_lines.append(line.strip())

writer_new_vocab = open('../pretrained-LMs/'+bert_model_name+'/vocab_dict.txt', 'w')
for line in list_lines:
    writer_new_vocab.write(line+' 1\n')
writer_new_vocab.close()
