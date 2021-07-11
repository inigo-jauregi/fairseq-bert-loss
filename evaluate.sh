DATASET_FOLDER=datasets/de-en_IWSLT2014/data
PREP_TEST=good
REF=dev/dev.tok.clean.lwc.bpe.en
SEED_NUM=1_val_plots
MODEL=models/MAX_F_BERT_gumbel-softmax.CONVERGENCE_LR_5e-5/seed_$SEED_NUM
EPOCH=_best.$PREP_TEST

# Obtain predictions hhh
grep ^H $DATASET_FOLDER/$MODEL/baseline.txt | cut -f3 > $DATASET_FOLDER/$MODEL/hypothesis$EPOCH.txt


# Re-order predictions
python scripts/re_order_preds.py $DATASET_FOLDER/$MODEL/baseline.txt $DATASET_FOLDER/$MODEL/hypothesis$EPOCH.txt $DATASET_FOLDER/$MODEL/hypothesis.ord.$EPOCH.txt
  

# Compute BLEU normal
perl mosesdecoder/scripts/generic/multi-bleu.perl $DATASET_FOLDER/$REF < $DATASET_FOLDER/$MODEL/hypothesis.ord.$EPOCH.txt > $DATASET_FOLDER/$MODEL/BLEU$EPOCH.kafuti.txt

# Compute BERT scores
python scripts/bert_score.py bert-base-uncased $DATASET_FOLDER/$MODEL/hypothesis.ord.$EPOCH.txt $DATASET_FOLDER/$REF $DATASET_FOLDER/$MODEL/BERT$EPOCH.kafuti.txt cuda:0

# Compute BERT score list
python scripts/bert_score_per_sentence.py bert-base-uncased $DATASET_FOLDER/$MODEL/hypothesis.ord.$EPOCH.txt $DATASET_FOLDER/$REF $DATASET_FOLDER/$MODEL/F_BERT_sentence.$EPOCH.kafuti.txt cuda:0
