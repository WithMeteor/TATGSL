import re
import os
import numpy as np
from tqdm import tqdm


def load_dataset(dataset_name):
    raw_path = './data/raw'
    with open(f"{raw_path}/{dataset_name}.texts.txt", "r", encoding="latin1") as file:
        text_list = file.read().strip().split("\n")
    with open(f"{raw_path}/{dataset_name}.labels.txt", "r") as file:
        label_list = file.read().strip().split("\n")
    return text_list, label_list


def filter_text(text: str):
    other_char = re.compile(r"[^A-Za-z0-9(),!?\'`]", flags=0)
    text = re.sub(other_char, " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", r" ( ", text)
    text = re.sub(r"\)", r" ) ", text)
    text = re.sub(r"\?", r" ? ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip().lower()


def main():

    save_path = './data/temp/'
    os.makedirs(save_path, exist_ok=True)

    for dataset in ['ohsumed', '20ng', 'R8', 'R52', 'AGNews', 'snippets']:

        print('Load dataset:', dataset)
        texts, labels = load_dataset(dataset)

        # handle labels
        label2index = {l: i for i, l in enumerate(sorted(set(labels)))}
        targets = [label2index[lb] for lb in labels]
        np.save(f"{save_path}/{dataset}.targets.npy", targets)

        # handle texts
        print('Filtering text...')
        texts_clean = []
        for t in tqdm(texts, ascii=True):
            texts_clean.append(filter_text(t))

        # save
        with open(f"{save_path}/{dataset}.texts.remove.txt", "w") as f:
            f.write("\n".join(texts_clean))


if __name__ == '__main__':
    main()
