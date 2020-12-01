
DATASET_FOLDER=datasets/de-en_TEDtalks
MODEL=models/BAS_TRANS_try
CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATASET_FOLDER/preprocess_bert_try --path $DATASET_FOLDER/$MODEL/checkpoints/checkpoint2.pt \
             --batch-size 128 --beam 5 > $DATASET_FOLDER/$MODEL/checkpoints/output2.txt
