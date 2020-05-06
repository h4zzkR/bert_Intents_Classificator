import pandas as pd
import numpy as np
import os
from pathlib import Path
from utils import parse_line


def encode_dataset(tokenizer, text_sequences, max_length):
    token_ids = np.zeros(shape=(len(text_sequences), max_length),
                         dtype=np.int32)
    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence)
        token_ids[i, 0:len(encoded)] = encoded
    attention_masks = (token_ids != 0).astype(np.int32)
    return {"input_ids": token_ids, "attention_masks": attention_masks}


def load_test_data(datadir='data'):
    lines_valid = Path(os.path.join(datadir, 'valid')).read_text().strip().splitlines()
    return pd.DataFrame([parse_line(line) for line in lines_valid])

def load_intents_map(datadir='data'):
    intent_names = Path(os.path.join(datadir, "vocab.intent")).read_text().split()
    intent_map = dict((label, idx) for idx, label in enumerate(intent_names))
    return intent_map


def load_prepare_dataset(datadir='data'):
    print('Loading data...')
    lines_train = Path(os.path.join(datadir, 'train')).read_text().strip().splitlines()
    lines_valid = Path(os.path.join(datadir, 'valid')).read_text().strip().splitlines()
    lines_test = Path(os.path.join(datadir, 'test')).read_text().strip().splitlines()

    parsed = [parse_line(line) for line in lines_train]

    df_train = pd.DataFrame([p for p in parsed if p is not None])
    df_valid = pd.DataFrame([parse_line(line) for line in lines_valid])
    df_test = pd.DataFrame([parse_line(line) for line in lines_test])

    intent_names = Path(os.path.join(datadir, "vocab.intent")).read_text().split()
    intent_map = dict((label, idx) for idx, label in enumerate(intent_names))

    return df_train, df_valid, df_test, intent_names, intent_map
