
DATASET_FOLDER=datasets/de-en_IWSLT2014/data
MODEL=models/MAX_F_BERT_gumbel-softmax.CONVERGENCE_LR_5e-5
SEED_NUM=1_good
PREP_TEST=good
EPOCH=_best
CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATASET_FOLDER/prep_files/preprocess_$PREP_TEST \
             --path $DATASET_FOLDER/$MODEL/seed_$SEED_NUM/checkpoint$EPOCH.pt \
             --batch-size 128 --beam 5 --use-bert-model --bos [unused0] --pad [PAD] --eos [unused1] --unk [UNK] \
             --tgtdict_add_sentence_limit_words_after > $DATASET_FOLDER/$MODEL/seed_$SEED_NUM/output$EPOCH.$PREP_TEST.txt
