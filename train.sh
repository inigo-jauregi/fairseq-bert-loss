DATASET_FOLDER=datasets/de-en_TEDtalks
MODEL=models/BAS_TRANS_bert_tau_0.1_gumbel
CUDA_VISIBLE_DEVICES=0 fairseq-train $DATASET_FOLDER/preprocess_bert_try \
             --lr 0.0002 -s de -t en --optimizer adam --max-tokens 1024 --dropout 0.1 \
             --arch transformer_iwslt_de_en --save-dir $DATASET_FOLDER/$MODEL/checkpoints \
             --max-epoch 1 --lr-scheduler inverse_sqrt \
             --warmup-updates 4000 --warmup-init-lr '1e-07' --min-lr '1e-09' \
             --adam-betas "(0.9, 0.98)" --weight-decay 0.0001 \
             --criterion bert_loss --bert-model bert-base-uncased --tau-gumbel-softmax 0.1 \
             --use-bert-model --bos [unused0] --pad [PAD] --eos [unused1] --unk [UNK] \
             --tgtdict_add_sentence_limit_words_after

             #--criterion bert_loss --bert-model bert-base-uncased \
             #--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
