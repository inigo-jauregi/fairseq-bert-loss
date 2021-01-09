DATASET_FOLDER=datasets/de-en_IWSLT2014/data
PREP_TEST=good
REF=test/edunov_test.tok.clean.lwc.bpe.en
MODEL=models/MIXED_0.3_NLL_BERT_GUMBEL_TAU_0.1_SOFT_BERT_SCORE_CONVERGENCE/checkpoints
EPOCH=_best.$PREP_TEST

# Obtain predictions
grep ^H $DATASET_FOLDER/$MODEL/output$EPOCH.txt | cut -f3 > $DATASET_FOLDER/$MODEL/hypothesis$EPOCH.txt


# Re-order predictions
python scripts/re_order_preds.py $DATASET_FOLDER/$MODEL/output$EPOCH.txt $DATASET_FOLDER/$MODEL/hypothesis$EPOCH.txt $DATASET_FOLDER/$MODEL/hypothesis.ord.$EPOCH.txt
  

# Compute BLEU normal
perl mosesdecoder/scripts/generic/multi-bleu.perl $DATASET_FOLDER/$REF < $DATASET_FOLDER/$MODEL/hypothesis.ord.$EPOCH.txt > $DATASET_FOLDER/$MODEL/BLEU$EPOCH.txt

# Compute BERT scores
python scripts/bert_score.py bert-base-uncased $DATASET_FOLDER/$MODEL/hypothesis.ord.$EPOCH.txt $DATASET_FOLDER/$REF $DATASET_FOLDER/$MODEL/BERT$EPOCH.txt cuda:0
