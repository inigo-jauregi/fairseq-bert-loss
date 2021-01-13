import random

# Data
dataset = 'eu-en_TEDtalks'
type = 'train'
prefix = 'train'
src = 'eu'
tgt = 'en'

# Open files
src_file = open('../datasets/' + dataset + '/' + type + '/' + prefix + '.' + src)
tgt_file = open('../datasets/' + dataset + '/' + type + '/' + prefix + '.' + tgt)

src_sen_lists = []
for line in src_file:
    line_clean = line.strip()
    src_sen_lists.append(line_clean)

tgt_sen_lists = []
for line in tgt_file:
    line_clean = line.strip()
    tgt_sen_lists.append(line_clean)

print(len(src_sen_lists))
print(len(tgt_sen_lists))

validation_ids = random.sample(range(len(src_sen_lists)), 400)

new_train_src_file = open('../datasets/' + dataset + '/' + type + '/train_no_val.' + src, 'w')
new_train_tgt_file = open('../datasets/' + dataset + '/' + type + '/train_no_val.' + tgt, 'w')
dev_src_file = open('../datasets/' + dataset + '/train/dev.' + src, 'w')
dev_tgt_file = open('../datasets/' + dataset + '/train/dev.' + tgt, 'w')

for i in range(len(src_sen_lists)):
    if i in validation_ids:
        dev_src_file.write(src_sen_lists[i] + '\n')
        dev_tgt_file.write(tgt_sen_lists[i] + '\n')
    else:
        new_train_src_file.write(src_sen_lists[i] + '\n')
        new_train_tgt_file.write(tgt_sen_lists[i] + '\n')

new_train_src_file.close()
new_train_tgt_file.close()
dev_src_file.close()
dev_tgt_file.close()
