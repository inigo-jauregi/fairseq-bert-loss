
DATASET_FOLDER=datasets/de-en_IWSLT2014/data
MODEL=models/BASELINE_TRANS_convergence_fairseq
for SEED_NUM in 4; do
  MODEL=models/BASELINE_TRANS_convergence_fairseq
  CUDA_VISIBLE_DEVICES=0 fairseq-train $DATASET_FOLDER/prep_files/preprocess_good \
               --lr 5e-4 -s de -t en --optimizer adam --max-tokens 4096 --clip-norm 0.0 --dropout 0.3 \
               --arch transformer_iwslt_de_en --save-dir $DATASET_FOLDER/$MODEL/seed_$SEED_NUM \
               --lr-scheduler inverse_sqrt --no-epoch-checkpoints --no-last-checkpoints \
               --warmup-updates 4000 --warmup-init-lr '1e-07' --min-lr '1e-09' \
               --adam-betas "(0.9, 0.98)" --weight-decay 0.0001 \
               --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
               --use-bert-model --bos [unused0] --pad [PAD] --eos [unused1] --unk [UNK] \
               --tgtdict_add_sentence_limit_words_after --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --patience 3 --seed $SEED_NUM

done

             #--criterion bert_loss --bert-model bert-base-uncased --tau-gumbel-softmax 0.1 \
             #--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
