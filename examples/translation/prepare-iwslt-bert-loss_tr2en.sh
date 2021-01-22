#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.shjj

SCRIPTS=../mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=../subword-nmt/subword_nmt
BPE_TOKENS=32000
BERT_MODEL=pretrained-LMs/bert-base-uncased
BERT_MODEL_SRC=pretrained-LMs/dbmdz/bert-base-turkish-uncased

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=tr
tgt=en
dataset=datasets/tr-en_TEDtalks
dataset_out=datasets/tr-en_TEDtalks
train=train
dev=dev2010
test=test

echo "pre-processing train data..."
for l in $src; do
    f=$train.$l
    tok=$train.tok

    #cat $dataset/train/$f | \
    #perl $TOKENIZER -threads 8 -l $l > $dataset_out/train/$tok.$l
    python scripts/bert_tokenize.py $BERT_MODEL_SRC $dataset/train/$f $dataset_out/train/$tok.$l
    #Segment the Chinese part
    #python -m jieba -d ' ' < $dataset/train/$f > $dataset/train/$tok.$l
    echo ""
done
for l in $tgt; do
    f=$train.$l
    tok=$train.tok

    #cat $dataset/train/$f | \
    #perl $TOKENIZER -threads 8 -l $l > $dataset_out/train/$tok.$l
    #cp $dataset/train/$f $dataset_out/train/$tok.$l
    python scripts/bert_tokenize.py $BERT_MODEL $dataset/train/$f $dataset_out/train/$tok.$l
    #python -m jieba -d ' ' < $dataset/train/$f > $dataset_out/train/$tok.$l
    echo ""
done
perl $CLEAN -ratio 1.5 $dataset_out/train/$tok $src $tgt $dataset_out/train/$tok.clean 1 175

for l in $src $tgt; do
    perl $LC < $dataset_out/train/$tok.clean.$l > $dataset_out/train/$tok.clean.lwc.$l
done

echo "pre-processing dev data..."
for l in $src; do
    f=$dev.$l
    tok=$dev.tok

    #cat $dataset/dev/$f | \
    #perl $TOKENIZER -threads 8 -l $l > $dataset_out/dev/$tok.$l
    python scripts/bert_tokenize.py $BERT_MODEL_SRC $dataset/dev/$f $dataset_out/dev/$tok.$l
    #python -m jieba -d ' ' < $dataset/dev/$f > $dataset/dev/$tok.$l
    echo ""
done
for l in $tgt; do
    f=$dev.$l
    tok=$dev.tok

    #cat $dataset/dev/$f | \
    #perl $TOKENIZER -threads 8 -l $l > $dataset_out/dev/$tok.$l
    python scripts/bert_tokenize.py $BERT_MODEL $dataset/dev/$f $dataset_out/dev/$tok.$l
    #cp $dataset/dev/$f $dataset_out/dev/$tok.$l
    #python -m jieba -d ' ' < $dataset/dev/$f > $dataset_out/dev/$tok.$l
    echo ""
done
perl $CLEAN -ratio 1.5 $dataset_out/dev/$tok $src $tgt $dataset_out/dev/$tok.clean 1 175
for l in $src $tgt; do
    perl $LC < $dataset_out/dev/$tok.clean.$l > $dataset_out/dev/$tok.clean.lwc.$l
done

echo "pre-processing test data..."
for l in $src; do
    f=$test.$l
    tok=$test.tok

    #cat $dataset/test/$f | \
    #perl $TOKENIZER -threads 8 -l $l > $dataset_out/test/$tok.$l
    python scripts/bert_tokenize.py $BERT_MODEL_SRC $dataset/test/$f $dataset_out/test/$tok.$l
    #python -m jieba -d ' ' < $dataset/test/$f > $dataset/test/$tok.$l
    echo ""
done
for l in $tgt; do
    f=$test.$l
    tok=$test.tok

    #cat $dataset/test/$f | \
    #perl $TOKENIZER -threads 8 -l $l > $dataset_out/test/$tok.$l
    #cp $dataset/test/$f $dataset_out/test/$tok.$l
    python scripts/bert_tokenize.py $BERT_MODEL $dataset/test/$f $dataset_out/test/$tok.$l
    #python -m jieba -d ' ' < $dataset/test/$f > $dataset_out/test/$tok.$l
    echo ""
done
perl $CLEAN -ratio 1.5 $dataset_out/test/$tok $src $tgt $dataset_out/test/$tok.clean 1 175
for l in $src $tgt; do
    perl $LC < $dataset_out/test/$tok.clean.$l > $dataset_out/test/$tok.clean.lwc.$l
done


#echo "creating train, valid, test..."
#for l in $src $tgt; do
#    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/valid.$l
#    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/train.$l
#
#    cat $tmp/IWSLT14.TED.dev2010.de-en.$l \
#        $tmp/IWSLT14.TEDX.dev2012.de-en.$l \
#        $tmp/IWSLT14.TED.tst2010.de-en.$l \
#        $tmp/IWSLT14.TED.tst2011.de-en.$l \
#        $tmp/IWSLT14.TED.tst2012.de-en.$l \
#        > $tmp/test.$l
#done

TRAIN=$dataset_out/train/$tok.clean.lwc.not_joined
BPE_CODE=$dataset_out/train/code
rm -f $TRAIN
for l in $src; do
    cat $dataset_out/train/$train.tok.clean.lwc.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

#for L in $src; do
#    echo "apply_bpe.py to train ${L}..."
#    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $dataset_out/train/$train.tok.clean.lwc.$L > $dataset_out/train/$train.tok.clean.lwc.bpe.$L
#    echo "apply_bpe.py to dev ${L}..."
#    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $dataset_out/dev/$dev.tok.clean.lwc.$L > $dataset_out/dev/$dev.tok.clean.lwc.bpe.$L
#    echo "apply_bpe.py to test ${L}..."
#    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $dataset_out/test/$test.tok.clean.lwc.$L > $dataset_out/test/$test.tok.clean.lwc.bpe.$L
#done

for L in $src; do
    cp $dataset_out/train/$train.tok.clean.lwc.$L $dataset_out/train/$train.tok.clean.lwc.bpe.$L
    cp $dataset_out/dev/$dev.tok.clean.lwc.$L $dataset_out/dev/$dev.tok.clean.lwc.bpe.$L
    cp $dataset_out/test/$test.tok.clean.lwc.$L $dataset_out/test/$test.tok.clean.lwc.bpe.$L
done

#for L in $tgt; do
#    echo "Apply BERT tokenization to train ${L}..."
#    python scripts/bert_tokenize.py $BERT_MODEL $dataset_out/train/$train.tok.clean.lwc.$L $dataset_out/train/$train.tok.clean.lwc.bpe.$L
#    echo "Apply BERT tokenization to dev ${L}..."
#    python scripts/bert_tokenize.py $BERT_MODEL $dataset_out/dev/$dev.tok.clean.lwc.$L $dataset_out/dev/$dev.tok.clean.lwc.bpe.$L
#    echo "Apply BERT tokenization to test ${L}..."
#    python scripts/bert_tokenize.py $BERT_MODEL $dataset_out/test/$test.tok.clean.lwc.$L $dataset_out/test/$test.tok.clean.lwc.bpe.$L
#done

for L in $tgt; do
    #echo "Apply BERT tokenization to train ${L}..."
    cp $dataset_out/train/$train.tok.clean.lwc.$L $dataset_out/train/$train.tok.clean.lwc.bpe.$L
    #echo "Apply BERT tokenization to dev ${L}..."
    cp $dataset_out/dev/$dev.tok.clean.lwc.$L $dataset_out/dev/$dev.tok.clean.lwc.bpe.$L
    #echo "Apply BERT tokenization to test ${L}..."
    cp $dataset_out/test/$test.tok.clean.lwc.$L $dataset_out/test/$test.tok.clean.lwc.bpe.$L
done
