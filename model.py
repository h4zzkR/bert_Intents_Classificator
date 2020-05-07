import tensorflow as tf
from transformers import TFBertModel
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling1D
from data_preprocess import encode_dataset


class IntentClassificationModelBase(tf.keras.Model):

    def __init__(self, intent_num_labels=None, model_name="bert-base-cased",
                 dropout_prob=0.1):
        super().__init__(name="joint_intent_slot")
        self.bert = TFBertModel.from_pretrained(model_name)
        self.dropout = Dropout(dropout_prob)

        # Use the default linear activation (no softmax) to compute logits.
        # The softmax normalization will be computed in the loss function
        # instead of the model itself.
        self.intent_classifier = Dense(intent_num_labels)

    def call(self, inputs, **kwargs):
        sequence_output, pooled_output = self.bert(inputs, **kwargs)
        pooled_output = self.dropout(pooled_output,
                                     training=kwargs.get("training", False))
        
        intent_logits = self.intent_classifier(pooled_output)
        return intent_logits



class IntentClassificationModel():
    def __init__(self, tokenizer, ckp_path, intents_map, load=True):
        self.tokenizer = tokenizer
        self.intents_map = intents_map
        self.model = IntentClassificationModelBase(len(intents_map.keys()))
        self.model.load_weights(ckp_path)

    def classify(self, text, map_intent=True):
        # is it works?
        inputs = tf.constant(self.tokenizer.encode(text))[None, :]  # batch_size = 1
        class_id = self.model(inputs).numpy().argmax(axis=1)[0]
        # print(class_id, self.intents_map)
        return self.intents_map[class_id]

    def encode(self, text_sequence, max_length):
        return encode_dataset(self.tokenizer, text_sequence, max_length)




