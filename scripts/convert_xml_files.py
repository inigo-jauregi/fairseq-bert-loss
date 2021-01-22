# Data
dataset = 'en-ru_TEDtalks'
type = 'test'
year = 'tst2012'
prefix = 'IWSLT14.TED.' + year + '.ru-en'
src = 'en'
tgt = 'ru'

# Open files
src_file = open('../datasets/' + dataset + '/' + type + '/' + prefix + '.' + src + '.xml')
tgt_file = open('../datasets/' + dataset + '/' + type + '/' + prefix + '.' + tgt + '.xml')

src_sen_lists = []
for line in src_file:
    line_clean = line.strip()
    if line_clean.startswith('<seg id'):
        line_clean = line_clean.replace('<seg id', '').replace('</seg>', '').split('>')[-1].strip()
        src_sen_lists.append(line_clean)

tgt_sen_lists = []
for line in tgt_file:
    line_clean = line.strip()
    if line.startswith('<seg id'):
        line_clean = line_clean.replace('<seg id', '').replace('</seg>', '').split('>')[-1].strip()
        tgt_sen_lists.append(line_clean)

print(len(src_sen_lists))
print(len(tgt_sen_lists))
assert len(src_sen_lists) == len(tgt_sen_lists), "Different number of sentences!"

# Write file
src_file_write = open('../datasets/' + dataset + '/' + type + '/' + year + '.' + src, 'w')
tgt_file_write = open('../datasets/' + dataset + '/' + type + '/' + year + '.' + tgt, 'w')

for i in range(len(src_sen_lists)):
    src_file_write.write(src_sen_lists[i] + '\n')
    tgt_file_write.write(tgt_sen_lists[i] + '\n')

src_file_write.close()
tgt_file_write.close()
