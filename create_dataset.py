import random
from transformers import AutoTokenizer
import pandas as pd

INPUT_TXT_FILE = 'PATH/TO/FILE.TXT'
OUTPUT_DIR = 'PATH/TO/DIR'
PARSBERT_TOKENIZER = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
SPECIAL_LABELS = {'،': 'I-COMMA',
                  '.': 'I-DOT',
                  '؟': 'I-QMARK',
                  '؛': 'I-SEMICOLON',
                  ':': 'I-COLON',
                  '!': 'I-EXCLAMATION'}
NORMAL_LABEL = 'O'


def _discrete_and_label(list_of_lines):
    list_of_lists = []
    for i, line in enumerate(list_of_lines):
        tokenized_line = PARSBERT_TOKENIZER.tokenize(line)
        if len(tokenized_line) < 50:
            continue
        for word in tokenized_line:
            lbl = NORMAL_LABEL
            brk = False
            sl = SPECIAL_LABELS.get(word, None)
            if sl:
                if list_of_lists:
                    list_of_lists[-1][2] = sl
                    brk = True
            if not brk:
                list_of_lists.append([i, word, lbl])
    return list_of_lists


def _shuffle_dataset(pd_dataset):
    groups = [df for _, df in pd_dataset.groupby('sentence_id')]
    random.shuffle(groups)
    shuffled_dataset = pd.concat(groups).reset_index(drop=True)
    return shuffled_dataset


def _get_train_val_test(dataset, percentage=[80, 10, 10]):
    assert len(percentage) == 3
    assert percentage[0] + percentage[1] + percentage[2] == 100
    all_numbers = dataset.drop_duplicates('sentence_id').reset_index(drop=True)
    one = int(len(all_numbers) * percentage[0] / 100)
    two = int(len(all_numbers) * (percentage[0] + percentage[1]) / 100)
    first = int(all_numbers[one:one + 1]['sentence_id'])
    second = int(all_numbers[two:two + 1]['sentence_id'])
    val_from = dataset[dataset.sentence_id == first].first_valid_index()
    train_to = val_from - 1
    test_from = dataset[dataset.sentence_id == second].first_valid_index()
    val_to = test_from - 1
    train_df = dataset[:train_to]
    val_df = dataset[val_from:val_to]
    test_df = dataset[test_from:]
    return train_df, val_df, test_df


def run():
    with open(f"{INPUT_TXT_FILE}", 'r') as reader:
        spc = reader.readlines()
        first_filtered = [i.replace("-", ' ').replace("(", " ").replace(")", " ").replace("»", " ").replace("«", " ").replace("\n", " ").replace("\t", " ") for i in spc]  # noqa
        del spc
    dataset = _discrete_and_label(first_filtered)
    pd_dataset = pd.DataFrame(dataset, columns=["sentence_id", "words", "labels"])

    shuffled_dataset = _shuffle_dataset(pd_dataset)
    del pd_dataset
    shuffled_dataset.to_csv(f'{OUTPUT_DIR}/whole_dataset.csv')

    train, val, test = _get_train_val_test(shuffled_dataset)
    del shuffled_dataset
    train.to_csv(f'{OUTPUT_DIR}/train.csv')
    val.to_csv(f'{OUTPUT_DIR}/val.csv')
    test.to_csv(f'{OUTPUT_DIR}/test.csv')
