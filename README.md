
--------------------------------------------------------------------------------

# Requirements

* torch >= 1.6.0
* transformers == 3.4.0
* Python version >= 3.7
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

```bash
pip install -r requirements.txt
```

# Quick Start

## Preprocessing the data

1 - Download the data (e.g. IWSLT data) and divide it into train, dev and test sets.

2 - Run the preprocessing script ```examples/translation/prepare-iwslt-bert-loss.sh```. 
Change the file paths as required.

3 - Prepare the files for training:

```bash
python fairseq_cli/preprocess.py \
    --source-lang <src_lang> \
    --target-lang <tgt_lang> \
    --trainpref <path_to_training_files> \
    --validpref <path_to_validation_files> \
    --testpref <path_to_test_files> \
    --destdir <path_to_save_preprocessed_files> \
    --tgtdict <pretrained_LMs_vocabulary> \
    --bos [unused0] \
    --pad [PAD] \
    --eos [unused1] \
    --unk [UNK] \
    --tgtdict_add_sentence_limit_words_after
```

4 - Model training (with recommended hyperparameters):

The ```--marginalization``` argument allows for these values: 
```raw``` (dense vectors), ```sparsemax``` or ```gumbel-softmax```.

```bash
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/train.py \
    <path_to_preprocessed_files> \
    --lr 5e-5 \
    -s de \
    -t en \
    --optimizer adam \
    --max-tokens 4096 \
    --clip-norm 0.0 \
    --dropout 0.3 \
    --arch transformer_iwslt_de_en \
    --save-dir <path_to_save_model> \
    --lr-scheduler inverse_sqrt \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --warmup-updates 4000 \
    --warmup-init-lr '1e-07' \
    --min-lr '1e-09' \
    --adam-betas "(0.9, 0.98)" \
    --weight-decay 0.0001 \
    --criterion aligned_bert_loss \
    --bert-model <path_to_LM> \
    --marginalization gumbel-softmax \
    --tau-gumbel-softmax 0.1 \
    --use-bert-model \
    --bos [unused0] \
    --pad [PAD] \
    --eos [unused1] \
    --unk [UNK] \
    --tgtdict_add_sentence_limit_words_after \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --patience 3 \
    --seed 1 \
    --finetune-from-model <pretrained_transformer>
```

5 - Inference:

```bash
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/generate.py \
    <path_to_preprocessed_files \
    --path <path_to_trained_model> \
    --batch-size 128 \
    --beam 5 \
    --use-bert-model \
    --bos [unused0] \
    --pad [PAD] \
    --eos [unused1] \
    --unk [UNK] \
    --tgtdict_add_sentence_limit_words_after
```
