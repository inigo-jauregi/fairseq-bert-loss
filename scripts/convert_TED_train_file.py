# Data
dataset = 'de-en_IWSLT2014/data'
type = 'train'
prefix = 'train.tags.de-en'
src = 'de'
tgt = 'en'

# Open files
src_file = open('../datasets/' + dataset + '/' + type + '/' + prefix + '.' + src)
tgt_file = open('../datasets/' + dataset + '/' + type + '/' + prefix + '.' + tgt)

src_sen_lists = []
for line in src_file:
    line_clean = line.strip()
    if not line_clean.startswith('<'):
        src_sen_lists.append(line_clean)

tgt_sen_lists = []
for line in tgt_file:
    line_clean = line.strip()
    if not line.startswith('<'):
        tgt_sen_lists.append(line_clean)

assert len(src_sen_lists) == len(tgt_sen_lists), "Different number of sentences!"

# Write file
src_file_write = open('../datasets/' + dataset + '/' + type + '/' + type + '.' + src, 'w')
tgt_file_write = open('../datasets/' + dataset + '/' + type + '/' + type + '.' + tgt, 'w')

for i in range(len(src_sen_lists)):
    src_file_write.write(src_sen_lists[i] + '\n')
    tgt_file_write.write(tgt_sen_lists[i] + '\n')

src_file_write.close()
tgt_file_write.close()
