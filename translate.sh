
DATASET_FOLDER=datasets/de-en_IWSLT2014/data
MODEL=models/MIXED_NLL_BERT_GUMBEL_TAU_HARD_CONVERGENCE
PREP_TEST=good
EPOCH=_best
CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATASET_FOLDER/prep_files/preprocess_$PREP_TEST \
             --path $DATASET_FOLDER/$MODEL/checkpoints/checkpoint$EPOCH.pt \
             --batch-size 128 --beam 5 --use-bert-model --bos [unused0] --pad [PAD] --eos [unused1] --unk [UNK] \
             --tgtdict_add_sentence_limit_words_after > $DATASET_FOLDER/$MODEL/checkpoints/output$EPOCH.$PREP_TEST.txt
