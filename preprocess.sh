TEXT=datasets/de-en_TEDtalks
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train/train.tok --validpref $TEXT/dev/tst2012_tst2013.tok \
    --testpref $TEXT/test/tst2014_tst2015.tok \
    --destdir $TEXT
