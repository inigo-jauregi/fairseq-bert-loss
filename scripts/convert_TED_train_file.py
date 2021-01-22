# Data
dataset = 'en-ru_TEDtalks'
type = 'train'
prefix = 'train.tags.ru-en'
src = 'en'
tgt = 'ru'

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

src_sen_lists_pair = []
tgt_sen_lists_pair = []
for i in range(len(src_sen_lists)):
    if not src_sen_lists[i].startswith('<') and not tgt_sen_lists[i].startswith('<'):
        src_sen_lists_pair.append(src_sen_lists[i])
        tgt_sen_lists_pair.append(tgt_sen_lists[i])

print(len(src_sen_lists_pair))
print(len(tgt_sen_lists_pair))

assert len(src_sen_lists_pair) == len(tgt_sen_lists_pair), "Different number of sentences!"

# Write file
src_file_write = open('../datasets/' + dataset + '/' + type + '/' + type + '.' + src, 'w')
tgt_file_write = open('../datasets/' + dataset + '/' + type + '/' + type + '.' + tgt, 'w')

for i in range(len(src_sen_lists_pair)):
    src_file_write.write(src_sen_lists_pair[i] + '\n')
    tgt_file_write.write(tgt_sen_lists_pair[i] + '\n')

src_file_write.close()
tgt_file_write.close()
