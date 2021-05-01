TEXT=datasets/en-de_IWSLT2014/data
fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train/train_no_val.tok.clean.lwc.bpe --validpref $TEXT/dev/dev.tok.clean.lwc.bpe \
    --testpref $TEXT/test/edunov_test.tok.clean.lwc.bpe \
    --destdir $TEXT/prep_files/preprocess --tgtdict ./pretrained-LMs/dbmdz/bert-base-german-uncased/vocab_dict.txt \
    --bos unused0 --pad [PAD] --eos unused1 --unk [UNK] \
    --tgtdict_add_sentence_limit_words_after


