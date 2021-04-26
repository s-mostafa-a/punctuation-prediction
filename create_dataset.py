from transformers import AutoTokenizer
import pandas as pd

input_file = 'PATH/TO/FILE.TXT'
out_put_dir = 'PATH/TO/DIR'
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


def run():
    with open(f"{input_file}", 'r') as reader:
        spc = reader.readlines()
        first_filtered = [i.replace("-", ' ').replace("(", " ").replace(")", " ").replace("»", " ").replace("«", " ").replace("\n", " ").replace("\t", " ") for i in spc]  # noqa
        del spc
    dataset = _discrete_and_label(first_filtered)
    pd_dataset = pd.DataFrame(dataset, columns=["sentence_id", "words", "labels"])
    pd_dataset.to_csv(f'{out_put_dir}/dataset.csv')
