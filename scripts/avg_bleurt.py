import sys


def main(scores_path, output_path):

    # Predictions
    preds_file = open(scores_path)
    preds_list = []
    for line in preds_file:
        line = line.strip().replace('\n','')
        preds_list.append(float(line))
    preds_file.close()

    # Compute average
    avg_bleurt = float(sum(preds_list)) / len(preds_list)

    # Output
    output_file = open(output_path, 'w')
    output_file.write('Avg. BLEURT: ' + str(avg_bleurt))
    output_file.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])