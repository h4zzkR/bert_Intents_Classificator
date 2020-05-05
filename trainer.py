import pandas as pd
from pathlib import Path
import argparse
import os
import datetime
from data_preprocess import load_prepare_dataset, encode_dataset
from transformers import BertTokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from model import IntentClassificationModelBase


parser = argparse.ArgumentParser(description='Trainer for bert-based intent classificator')
parser.add_argument('--model-name', type=str, default='bert-base-cased')
parser.add_argument('--max_length', type=int, default=45, help='Max tokens in sequence')


args = parser.parse_args()


class ModelTrainer():
    def __init__(self, model_name, max_length):
        df_train, df_valid, df_test, intent_names, self.intent_map = load_prepare_dataset('data')
        # Y's:
        self.intent_train = df_train["intent_label"].map(self.intent_map).values
        self.intent_valid = df_valid["intent_label"].map(self.intent_map).values
        self.intent_test = df_test["intent_label"].map(self.intent_map).values

        tokenizer = BertTokenizer.from_pretrained(model_name)
        # X's:
        print('Encoding data...')
        self.encoded_train = encode_dataset(tokenizer, df_train["words"], max_length)
        self.encoded_valid = encode_dataset(tokenizer, df_valid["words"], max_length)
        self.encoded_test = encode_dataset(tokenizer, df_test["words"], max_length)

        self.intent_model = IntentClassificationModelBase(
                    intent_num_labels=len(self.intent_map)
                    )

        self.intent_model.compile(optimizer=Adam(learning_rate=3e-5, epsilon=1e-08),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=[SparseCategoricalAccuracy('accuracy')])

    def train(self, epochs, batch_size, model_save_dir='model'):
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        time = datetime.datetime.now()
        name = f"intents_cls_e{epochs}_bs{batch_size}"

        checkpoint_path = os.path.join(model_save_dir, name)
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)


        history = self.intent_model.fit(self.encoded_train, self.intent_train, epochs=epochs, batch_size=batch_size,
                           validation_data=(self.encoded_valid, self.intent_valid), callbacks=cp_callback)

        return self.intent_model


if __name__ == "__main__":
    trainer = ModelTrainer(args.model_name, args.max_length)
    model = trainer.train(epochs=2, batch_size=32); del trainer
    print(model)