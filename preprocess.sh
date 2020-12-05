TEXT=datasets/de-en_TEDtalks
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train/train.tok.clean.lwc.bpe --validpref $TEXT/dev/tst2012_tst2013.tok.clean.lwc.bpe \
    --testpref $TEXT/dev/tst2012_tst2013.tok.clean.lwc.bpe \
    --destdir $TEXT/preprocess_bert_try_dev --tgtdict ./pretrained-LMs/bert-base-uncased/vocab_dict.txt \
    --bos [unused0] --pad [PAD] --eos [unused1] --unk [UNK] \
    --tgtdict_add_sentence_limit_words_after
