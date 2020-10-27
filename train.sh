DATASET_FOLDER=datasets/de-en_TEDtalks
MODEL=models/BAS_TRANS
fairseq-train $DATASET_FOLDER/$MODEL/preprocessed_data \
    --lr 0.25 --optimizer adam --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch transformer_iwslt_de_en --save-dir $DATASET_FOLDER/$MODEL/checkpoints