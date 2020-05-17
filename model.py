import tensorflow as tf
from transformers import TFBertModel
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling1D
from data_preprocess import encode_dataset


class SlotIntentDetectorModelBase(tf.keras.Model):

    def __init__(self, intent_num_labels=None, slot_num_labels=None,
                 model_name="bert-base-cased", dropout_prob=0.1):
        super().__init__(name="joint_intent_slot")
        self.bert = TFBertModel.from_pretrained(model_name)
        self.dropout = Dropout(dropout_prob)
        self.intent_classifier = Dense(intent_num_labels,
                                       name="intent_classifier")
        self.slot_classifier = Dense(slot_num_labels,
                                     name="slot_classifier")

    def call(self, inputs, **kwargs):
        sequence_output, pooled_output = self.bert(inputs, **kwargs)

        # The first output of the main BERT layer has shape:
        # (batch_size, max_length, output_dim)
        sequence_output = self.dropout(sequence_output,
                                       training=kwargs.get("training", False))
        slot_logits = self.slot_classifier(sequence_output)

        # The second output of the main BERT layer has shape:
        # (batch_size, output_dim)
        # and gives a "pooled" representation for the full sequence from the
        # hidden state that corresponds to the "[CLS]" token.
        pooled_output = self.dropout(pooled_output,
                                     training=kwargs.get("training", False))
        intent_logits = self.intent_classifier(pooled_output)

        return slot_logits, intent_logits



class SlotIntentDetectorModel():
    def __init__(self, tokenizer, ckp_path, intents_map, slots_map, load=True):
        self.tokenizer = tokenizer
        self.intents_map = intents_map # id2intent
        self.slots_map = slots_map # id2slot
        self.model = SlotIntentDetectorModelBase(len(intents_map.keys()))
        self.model.load_weights(ckp_path)

    def decode_predictions(text, intent_id, slot_ids):
        info = {"intent": self.intents_map[intent_id]}
        collected_slots = {}
        active_slot_words = []
        active_slot_name = None
        for word in text.split():
            tokens = self.tokenizer.tokenize(word)
            current_word_slot_ids = slot_ids[:len(tokens)]
            slot_ids = slot_ids[len(tokens):]
            current_word_slot_name = self.slots_map[current_word_slot_ids[0]]
            if current_word_slot_name == "O":
                if active_slot_name:
                    collected_slots[active_slot_name] = " ".join(active_slot_words)
                    active_slot_words = []
                    active_slot_name = None
            else:
                # Naive BIO: handling: treat B- and I- the same...
                new_slot_name = current_word_slot_name[2:]
                if active_slot_name is None:
                    active_slot_words.append(word)
                    active_slot_name = new_slot_name
                elif new_slot_name == active_slot_name:
                    active_slot_words.append(word)
                else:
                    collected_slots[active_slot_name] = " ".join(active_slot_words)
                    active_slot_words = [word]
                    active_slot_name = new_slot_name
        if active_slot_name:
            collected_slots[active_slot_name] = " ".join(active_slot_words)
        info["slots"] = collected_slots
        return info

    def classify(self, text, map_intent=True):
        # is it works?
        inputs = tf.constant(self.tokenizer.encode(text))[None, :]  # batch_size = 1
        class_id = self.model(inputs).numpy().argmax(axis=1)[0]
        # print(class_id, self.intents_map)
        return self.intents_map[class_id]


    def nlu(text):
        inputs = tf.constant(tokenizer.encode(text))[None, :]  # batch_size = 1
        slot_logits, intent_logits = self.model(inputs)
        slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]
        intent_id = intent_logits.numpy().argmax(axis=-1)[0]

        return self.decode_predictions(text, intent_id, slot_ids)

    def encode(self, text_sequence, max_length):
        return encode_dataset(self.tokenizer, text_sequence, max_length)




