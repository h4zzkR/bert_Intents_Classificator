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


def load_test_data(curdir):
    lines_valid = Path(os.path.join(curdir, 'data/valid')).read_text().strip().splitlines()
    return pd.DataFrame([parse_line(line) for line in lines_valid])

def load_intents_map(curdir):
    intent_names = Path(os.path.join(curdir, "data/vocab.intent")).read_text().split()
    intent_map = dict((label, idx) for idx, label in enumerate(intent_names))
    return intent_names, intent_map

def load_slots_map(curdir):
    slot_names = Path(os.path.join(curdir, "data/vocab.slot")).read_text().strip().splitlines()
    slot_map = {}
    for label in slot_names:
        slot_map[label] = len(slot_map)
    return slot_map


def encode_token_labels(text_sequences, slot_names, tokenizer, slot_map,
                        max_length):
    encoded = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
    for i, (text_sequence, word_labels) in enumerate(
            zip(text_sequences, slot_names)):
        encoded_labels = []
        for word, word_label in zip(text_sequence.split(), word_labels.split()):
            tokens = tokenizer.tokenize(word)
            encoded_labels.append(slot_map[word_label])
            expand_label = word_label.replace("B-", "I-")
            if not expand_label in slot_map:
                expand_label = word_label
            encoded_labels.extend([slot_map[expand_label]] * (len(tokens) - 1))
        encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
    return encoded



def load_prepare_dataset(curdir):
    print('Loading data...')
    lines_train = Path(os.path.join(curdir, 'data/train')).read_text().strip().splitlines()
    lines_valid = Path(os.path.join(curdir, 'data/valid')).read_text().strip().splitlines()
    lines_test = Path(os.path.join(curdir, 'data/test')).read_text().strip().splitlines()

    parsed = [parse_line(line) for line in lines_train]

    df_train = pd.DataFrame([p for p in parsed if p is not None])
    df_valid = pd.DataFrame([parse_line(line) for line in lines_valid])
    df_test = pd.DataFrame([parse_line(line) for line in lines_test])

    intent_names, intent_map = load_intents_map(curdir)
    slot_map = load_slots_map(curdir)
    
    return df_train, df_valid, df_test, intent_names, intent_map, slot_map