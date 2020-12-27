DATASET_FOLDER=datasets/de-en_TEDtalks
REF=test/tst2014_tst2015.tok.clean.lwc.bpe.en
MODEL=models/BAS_BERT_tau_0.1_1_epoch_FT/checkpoints
EPOCH=_best.test

# Obtain predictions
grep ^H $DATASET_FOLDER/$MODEL/output$EPOCH.txt | cut -f3 > $DATASET_FOLDER/$MODEL/hypothesis$EPOCH.txt


# Re-order predictions
python scripts/re_order_preds.py $DATASET_FOLDER/$MODEL/output$EPOCH.txt $DATASET_FOLDER/$MODEL/hypothesis$EPOCH.txt $DATASET_FOLDER/$MODEL/hypothesis.ord.$EPOCH.txt
  

# Compute BLEU normal
perl mosesdecoder/scripts/generic/multi-bleu.perl $DATASET_FOLDER/$REF < $DATASET_FOLDER/$MODEL/hypothesis.ord.$EPOCH.txt > $DATASET_FOLDER/$MODEL/BLEU$EPOCH.txt

# Compute BERT scores
python scripts/bert_score.py bert-base-uncased $DATASET_FOLDER/$MODEL/hypothesis.ord.$EPOCH.txt $DATASET_FOLDER/$REF $DATASET_FOLDER/$MODEL/BERT$EPOCH.txt cuda:0
