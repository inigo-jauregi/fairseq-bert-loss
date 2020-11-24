
bert_model = 'bert-base-uncased'

vocab_file = open('../pretrained-LMs/'+bert_model+'/vocab.txt')
list_lines = []
for line in vocab_file:
    list_lines.append(line.strip())

writer_new_vocab = open('../pretrained-LMs/'+bert_model+'/vocab_dict.txt', 'w')
for line in list_lines:
    writer_new_vocab.write(line+' 1\n')
writer_new_vocab.close()
