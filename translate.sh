
DATASET_FOLDER=datasets/de-en_TEDtalks
MODEL=models/BAS_TRANS_bert_hard_gumbel
EPOCH=1
CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATASET_FOLDER/preprocess_bert_try_dev \
             --path $DATASET_FOLDER/$MODEL/checkpoints/checkpoint$EPOCH.pt \
             --batch-size 128 --beam 5 --use-bert-model --bos [unused0] --pad [PAD] --eos [unused1] --unk [UNK] \
             --tgtdict_add_sentence_limit_words_after > $DATASET_FOLDER/$MODEL/checkpoints/output$EPOCH.dev.txt
