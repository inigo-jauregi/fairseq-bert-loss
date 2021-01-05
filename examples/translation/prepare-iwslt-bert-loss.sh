#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

SCRIPTS=../mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=../subword-nmt/subword_nmt
BPE_TOKENS=32000
BERT_MODEL=pretrained-LMs/bert-base-uncased

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de
tgt=en
dataset=datasets/de-en_IWSLT2014/data
train=train_no_val
dev=dev
test=dev2012

#echo "pre-processing train data..."
#for l in $src; do
#    f=$train.$l
#    tok=$train.tok

#    cat $dataset/train/$f | \
#    perl $TOKENIZER -threads 8 -l $l > $dataset/train/$tok.$l
#    echo ""
#done
#for l in $tgt; do
#    f=$train.$l
#    tok=$train.tok

#    cp $dataset/train/$f $dataset/train/$tok.$l
#    echo ""
#done
#perl $CLEAN -ratio 1.5 $dataset/train/$tok $src $tgt $dataset/train/$tok.clean 1 175
#for l in $src $tgt; do
#    perl $LC < $dataset/train/$tok.clean.$l > $dataset/train/$tok.clean.lwc.$l
#done

#echo "pre-processing dev data..."
#for l in $src $tgt; do
#    f=$dev.$l
#    tok=$dev.tok

#    cat $dataset/dev/$f | \
#    perl $TOKENIZER -threads 8 -l $l > $dataset/dev/$tok.$l
#    echo ""
#done
#for l in $tgt; do
#    f=$dev.$l
#    tok=$dev.tok

#    cp $dataset/dev/$f $dataset/dev/$tok.$l
#    echo ""
#done
#perl $CLEAN -ratio 1.5 $dataset/dev/$tok $src $tgt $dataset/dev/$tok.clean 1 175
#for l in $src $tgt; do
#    perl $LC < $dataset/dev/$tok.clean.$l > $dataset/dev/$tok.clean.lwc.$l
#done

echo "pre-processing test data..."
for l in $src $tgt; do
    f=$test.$l
    tok=$test.tok

    cat $dataset/test/$f | \
    perl $TOKENIZER -threads 8 -l $l > $dataset/test/$tok.$l
    echo ""
done
for l in $tgt; do
    f=$test.$l
    tok=$test.tok

    cp $dataset/test/$f $dataset/test/$tok.$l
    echo ""
done
perl $CLEAN -ratio 1.5 $dataset/test/$tok $src $tgt $dataset/test/$tok.clean 1 175
for l in $src $tgt; do
    perl $LC < $dataset/test/$tok.clean.$l > $dataset/test/$tok.clean.lwc.$l
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

TRAIN=$dataset/train/$tok.clean.lwc.not_joined
BPE_CODE=$dataset/train/code
#rm -f $TRAIN
#for l in $src; do
#    cat $dataset/train/$train.tok.clean.lwc.$l >> $TRAIN
#done

#echo "learn_bpe.py on ${TRAIN}..."
#python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src; do
    #echo "apply_bpe.py to train ${L}..."
    #python $BPEROOT/apply_bpe.py -c $BPE_CODE < $dataset/train/$train.tok.clean.lwc.$L > $dataset/train/$train.tok.clean.lwc.bpe.$L
    #echo "apply_bpe.py to dev ${L}..."
    #python $BPEROOT/apply_bpe.py -c $BPE_CODE < $dataset/dev/$dev.tok.clean.lwc.$L > $dataset/dev/$dev.tok.clean.lwc.bpe.$L
    echo "apply_bpe.py to test ${L}..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $dataset/test/$test.tok.clean.lwc.$L > $dataset/test/$test.tok.clean.lwc.bpe.$L
done

for L in $tgt; do
    #echo "Apply BERT tokenization to train ${L}..."
    #python scripts/bert_tokenize.py $BERT_MODEL $dataset/train/$train.tok.clean.lwc.$L $dataset/train/$train.tok.clean.lwc.bpe.$L
    #echo "Apply BERT tokenization to dev ${L}..."
    #python scripts/bert_tokenize.py $BERT_MODEL $dataset/dev/$dev.tok.clean.lwc.$L $dataset/dev/$dev.tok.clean.lwc.bpe.$L
    echo "Apply BERT tokenization to test ${L}..."
    python scripts/bert_tokenize.py $BERT_MODEL $dataset/test/$test.tok.clean.lwc.$L $dataset/test/$test.tok.clean.lwc.bpe.$L
done
