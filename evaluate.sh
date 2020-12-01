DATASET_FOLDER=datasets/de-en_TEDtalks
MODEL=models/BAS_TRANS_try/checkpoints

# Obtain target
grep ^T $DATASET_FOLDER/$MODEL/output2.txt | cut -f2 > $DATASET_FOLDER/$MODEL/target2.txt

# Obtain predictions
grep ^H $DATASET_FOLDER/$MODEL/output2.txt | cut -f3 > $DATASET_FOLDER/$MODEL/hypothesis2.txt

  

# Compute BLEU normal
perl mosesdecoder/scripts/generic/multi-bleu.perl $DATASET_FOLDER/$MODEL/target2.txt < $DATASET_FOLDER/$MODEL/hypothesis2.txt > $DATASET_FOLDER/$MODEL/BLEU2.txt

  
