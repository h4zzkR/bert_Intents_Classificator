import pandas as pd
from pathlib import Path
import argparse

# disable warnings from tf
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
from tqdm import tqdm
import datetime
from data_preprocess import load_prepare_dataset, encode_dataset
from transformers import BertTokenizer
from model import IntentClassificationModel
from data_preprocess import load_intents_map, load_test_data


def init(ckp_path, model_name="bert-base-cased"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    intents_map = load_intents_map()
    id2intent = {intents_map[i] : i for i in intents_map.keys()}
    model = IntentClassificationModel(tokenizer, ckp_path, id2intent)
    return intents_map, id2intent, model

def test_data_load(model, intents_map, max_length=46):
    df_test = load_test_data()
    intents_test = df_test["intent_label"].values
    return df_test["words"], intents_test

def test_model(model, intent2id):
    print('Testing model on valid data...')
    trues = 0
    test_data, labels_test = test_data_load(model, intent2id)
    for i, seq in enumerate(tqdm(test_data)):
        # print(seq); sys.exit()
        predict_label = model.classify(seq)
        true_label = labels_test[i]
        trues += (true_label == predict_label)
    i += 1
    print(f'Accuracy on test (valid data) is {round(trues/i, 3)}')

def main(ckp_path, test=True):
    intent2id, id2intent, model = init(ckp_path)
    if test:
        test_model(model, intent2id)

if __name__ == "__main__":
    ckp = '/home/samsepi00/lab/intent_detection/model/intents_cls_e2_bs32'
    main(ckp)