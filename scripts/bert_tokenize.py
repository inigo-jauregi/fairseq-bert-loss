import sys
from tqdm import tqdm
from transformers import AutoTokenizer


def main(bert_model, input_path, output_path):

    print(bert_model)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)

    input_file = open(input_path)
    output_file = open(output_path, 'w')
    for line in tqdm(input_file):
        line = line.strip()
        line_tokenized = " ".join(bert_tokenizer.tokenize(line))
        output_file.write(line_tokenized+'\n')

    input_file.close()
    output_file.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
