import os
import time
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers

def _tf_get_model(pretrained_name):
    if "roberta" in pretrained_name:
        mapping = transformers.TFRobertaForSequenceClassification
    elif "distilbert" in pretrained_name:
        mapping = transformers.TFDistilBertForSequenceClassification
    elif "albert" in pretrained_name:
        mapping = transformers.TFAlbertForSequenceClassification
    elif "bert" in pretrained_name:
        mapping = transformers.TFBertForSequenceClassification
    elif "xlm" in pretrained_name:
        mapping = transformers.TFXLMForSequenceClassification
    elif "xlnet" in pretrained_name:
        mapping = transformers.TFXLNetForSequenceClassification
    else:
        raise NotImplementedError
    return mapping


def create_model(model_name, num_labels=2):
    config = transformers.AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = _tf_get_model(model_name).from_pretrained(model_name, config=config)
    return model
