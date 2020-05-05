import tensorflow as tf
from transformers import TFBertModel
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling1D


class IntentClassificationModel(tf.keras.Model):

    def __init__(self, tokenizer,
                intent_num_labels=None, model_name="bert-base-cased",
                dropout_prob=0.1):
        super().__init__(name="joint_intent_slot")
        self.bert = TFBertModel.from_pretrained(model_name)
        
        print('Bert module initialized')
        self.dropout = Dropout(dropout_prob)

        # Use the default linear activation (no softmax) to compute logits.
        # The softmax normalization will be computed in the loss function
        # instead of the model itself.

        self.intent_classifier = Dense(intent_num_labels)
        self.tokenizer = tokenizer

    def call(self, inputs, **kwargs):
        sequence_output, pooled_output = self.bert(inputs, **kwargs)
        pooled_output = self.dropout(pooled_output,
                                     training=kwargs.get("training", False))
        
        intent_logits = self.intent_classifier(pooled_output)
        return intent_logits

    def classify(text, intent_names):
        # is it works?
        inputs = tf.constant(self.tokenizer.encode(text))[None, :]  # batch_size = 1
        class_id = self.predict(inputs).numpy().argmax(axis=1)[0]
        return intent_names[class_id]


