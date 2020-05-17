import pandas as pd
from pathlib import Path
import argparse

# disable warnings from tf
#   rename this shit
#   refactor this shit
import logging, os
# logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
from tqdm import tqdm
import datetime
from data_preprocess import load_prepare_dataset, encode_dataset
from transformers import BertTokenizer
from model import SlotIntentDetectorModel
from data_preprocess import load_intents_map, load_test_data, load_slots_map


def init(curdir, ckp_path, model_name="bert-base-cased"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    intents_names, intents_map = load_intents_map(curdir)
    slot_names, slot_map = load_slots_map(curdir)

    id2intent = {intents_map[i] : i for i in intents_map.keys()}
    id2slot = {slot_map[i] : i for i in slot_map.keys()}
    model = SlotIntentDetectorModel(tokenizer, ckp_path, id2intent, id2slot)
    return model, id2intent, id2slot

def test_data_load(curdir, model, intents_map, max_length=46):
    df_test = load_test_data(curdir)
    intents_test = df_test["intent_label"].values
    return df_test["words"], intents_test

def check_model(curdir):
    p = os.path.join(curdir, 'model/')
    ckp = Path(os.path.join(p, 'checkpoint')).read_text()
    m = ckp.splitlines()[0].split(': ')[-1]
    p = os.path.join(p, m[1:-1])
    return p

def main(test=False):
    # print('GPU_____:', tf.test.gpu_device_name())
    curdir = Path(__file__).parent.absolute()

    ckp_path = check_model(curdir)
    model, id2intent, id2slot = init(curdir, ckp_path)
    return model

if __name__ == "__main__":
    main()
