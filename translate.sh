
DATASET_FOLDER=datasets/de-en_IWSLT2014/data
BASELINE=models/BASELINE_TRANS_convergence_fairseq
MODEL=models/MAX_F_BERT_gumbel-softmax.CONVERGENCE_LR_5e-5
SEED_NUM=seed_1_val_plots
PREP_TEST=dev
EPOCH=_best
#for check in checkpoint_1_500.pt checkpoint_1_600.pt checkpoint_1_700.pt checkpoint_1_800.pt checkpoint_1_900.pt checkpoint_2_1000.pt checkpoint_2_1100.pt checkpoint_2_1200.pt checkpoint_2_1300.pt checkpoint_2_1400.pt checkpoint_2_1500.pt checkpoint_2_1600.pt checkpoint_2_1700.pt checkpoint_2_1800.pt checkpoint_2_1800.pt checkpoint_2_1900.pt checkpoint_3_2000.pt checkpoint_3_2100.pt checkpoint_3_2200.pt checkpoint_3_2300.pt checkpoint_3_2400.pt checkpoint_3_2500.pt checkpoint_3_2600.pt checkpoint_3_2700.pt checkpoint_3_2800.pt checkpoint_3_2900.pt checkpoint_4_3000.pt checkpoint_4_3100.pt checkpoint_4_3200.pt checkpoint_4_3300.pt checkpoint_4_3400.pt checkpoint_4_3500.pt checkpoint_4_3600.pt checkpoint_4_3700.pt checkpoint_4_3800.pt checkpoint_5_3900.pt checkpoint_5_4000.pt checkpoint_5_4100.pt checkpoint_5_4200.pt checkpoint_5_4300.pt checkpoint_5_4400.pt checkpoint_5_4500.pt; do
CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATASET_FOLDER/prep_files/preprocess_$PREP_TEST \
               --path $DATASET_FOLDER/$BASELINE/seed_1/checkpoint_best.pt \
               --batch-size 128 --beam 5 --use-bert-model --bos [unused0] --pad [PAD] --eos [unused1] --unk [UNK] \
               --tgtdict_add_sentence_limit_words_after > $DATASET_FOLDER/$MODEL/$SEED_NUM/baseline.txt
#done
