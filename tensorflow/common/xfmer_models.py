import os
import time
import tensorflow as tf
import tensorflow_datasets as tfds


def _tf_get_model(pretrained_name):
    if "roberta" in pretrained_name:
        mapping = xfmers.TFRobertaForSequenceClassification
    elif "distilbert" in pretrained_name:
        mapping = xfmers.TFDistilBertForSequenceClassification
    elif "albert" in pretrained_name:
        mapping = xfmers.TFAlbertForSequenceClassification
    elif "bert" in pretrained_name:
        mapping = xfmers.TFBertForSequenceClassification
    elif "xlm" in pretrained_name:
        mapping = xfmers.TFXLMForSequenceClassification
    elif "xlnet" in pretrained_name:
        mapping = xfmers.TFXLNetForSequenceClassification
    else:
        raise NotImplementedError
    return mapping


def create_model(model_name, num_labels=2):
    config = xfmers.AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = _tf_get_model(model_name).from_pretrained(model_name, config=config)
    return model


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
