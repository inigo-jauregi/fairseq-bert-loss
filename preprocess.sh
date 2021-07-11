TEXT=datasets/de-en_IWSLT2014/data
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train/train_no_val.tok.clean.lwc.bpe --validpref $TEXT/dev/dev.tok.clean.lwc.bpe \
    --testpref $TEXT/dev/dev.tok.clean.lwc.bpe \
    --destdir $TEXT/prep_files/preprocess_dev --tgtdict ./pretrained-LMs/bert-base-uncased/vocab_dict.txt \
    --bos [unused0] --pad [PAD] --eos [unused1] --unk [UNK] \
    --tgtdict_add_sentence_limit_words_after


