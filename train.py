from simpletransformers.ner import NERModel, NERArgs
from .create_dataset import SPECIAL_LABELS, NORMAL_LABEL, OUTPUT_DIR
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

LABEL_SET = list(SPECIAL_LABELS.keys()) + [NORMAL_LABEL]
NUMBER_OF_TRAIN_EPOCHS = 5


def _get_model(path_to_model_checkpoint=None):
    # path_to_model_checkpoint is path to checkpoint directory,
    # # something like this:
    # # # path_to_model_checkpoint = '/gdrive/MyDrive/training_output/checkpoint-22638-epoch-1/'
    model_args = NERArgs()
    model_args.overwrite_output_dir = True
    model_args.save_steps = -1
    if path_to_model_checkpoint is None:
        model = NERModel('bert',
                         'HooshvareLab/bert-fa-base-uncased',
                         labels=LABEL_SET,
                         args=model_args,
                         use_cuda=True)
    else:
        model = NERModel('bert',
                         path_to_model_checkpoint,
                         labels=LABEL_SET,
                         args=model_args,
                         use_cuda=True)
    return model


def _train_one_epoch(model):
    train_dataset = pd.read_csv(f'{OUTPUT_DIR}/train.csv').dropna()
    model.train_model(train_dataset, output_dir=f'{OUTPUT_DIR}')
    del train_dataset
    return model


def _eval_model(model, file='val'):
    # or file='test'
    test_dataset = pd.read_csv(f'{OUTPUT_DIR}/{file}.csv').dropna()
    groups = [df for _, df in test_dataset.groupby('sentence_id')]
    result, model_outputs, predictions = model.eval_model(test_dataset)
    trues = []
    for i, sentence in enumerate(predictions):
        trues.append(groups[i]['labels'].to_list()[:len(sentence)])

    predictions = list(itertools.chain.from_iterable(predictions))
    trues = list(itertools.chain.from_iterable(trues))
    print(classification_report(trues, predictions))
    return trues, predictions


def _create_confusion_matrix(trues, predictions):
    cm = confusion_matrix(trues, predictions)
    sums = cm.sum(axis=1)
    sums = np.expand_dims(sums, axis=1)
    nor_cm = cm / sums
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(nor_cm,
                ax=ax,
                annot=True,
                cmap='Blues')
    plt.yticks(rotation=0, size=12)
    plt.xticks(size=12)
    ax.set_xlabel('Predicted Label', size=14)
    ax.set_ylabel('True Label', size=14)


def run():
    model = _get_model()
    for i in range(NUMBER_OF_TRAIN_EPOCHS):
        model = _train_one_epoch(model)
        trues, predictions = _eval_model(model)
        _create_confusion_matrix(trues, predictions)
        del trues, predictions
