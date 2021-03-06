import time
import tensorflow.compat.v2 as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.applications as models

from . import mlperf_rn50


def toy_cnn(img_size=(224,224), num_class=2, weights=None, dtype=tf.float32):
    model = tf.keras.models.Sequential([
        layers.Conv2D(64, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu", input_shape=(img_size[0],img_size[1],3), dtype=dtype),
        layers.Conv2D(64, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.MaxPooling2D((4,4)),
        layers.BatchNormalization(fused=True),
        layers.Conv2D(128, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(128, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.MaxPooling2D((4,4)),
        layers.BatchNormalization(fused=True),
        layers.Conv2D(64, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(64, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.BatchNormalization(fused=True),
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_class),
        layers.Activation("softmax", dtype=tf.float32)
    ])
    return model

def huge_cnn(img_size=(224,224), num_class=2, weights=None, dtype=tf.float32):
    model = tf.keras.models.Sequential([
        layers.Conv2D(128, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu", input_shape=(img_size[0],img_size[1],3), dtype=dtype),
        layers.MaxPooling2D((4,4)),
        layers.BatchNormalization(fused=True),
        layers.Conv2D(512, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.MaxPooling2D((4,4)),
        layers.BatchNormalization(fused=True),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.MaxPooling2D((4,4)),
        layers.BatchNormalization(fused=True),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.Conv2D(1024, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
        layers.BatchNormalization(fused=True),
        layers.GlobalAveragePooling2D(),
        layers.Dense(512),
        layers.Dense(num_class),
        layers.Activation("softmax", dtype=tf.float32)
    ])
    return model


def rn50(img_size=(224,224), num_class=2, weights="imagenet", dtype=tf.float32):
    input_layer = layers.Input(shape=(img_size[0],img_size[1],3), dtype=dtype)
    base = models.ResNet50V2(input_tensor=input_layer, include_top=False, weights=weights)
    base.trainable = True
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_class)(x)
    preds = layers.Activation("softmax", dtype=tf.float32)(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=preds)
    return model

  
def rn50_mlperf(img_size=(224,224), num_class=2):
    model = mlperf_rn50.rn50(num_class, input_shape=(img_size[0], img_size[1], 3), batch_size=None, use_l2_regularizer=True)
    return model
  

def rn152(img_size=(224,224), num_class=2, weights="imagenet", dtype=tf.float32):
    input_layer = layers.Input(shape=(img_size[0],img_size[1],3), dtype=dtype)
    base = models.ResNet152V2(input_tensor=input_layer, include_top=False, weights=weights)
    base.trainable = True
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_class)(x)
    preds = layers.Activation("softmax", dtype=tf.float32)(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=preds)
    return model


def dn201(img_size=(224,224), num_class=2, weights="imagenet", dtype=tf.float32):
    input_layer = layers.Input(shape=(img_size[0],img_size[1],3), dtype=dtype)
    base = models.DenseNet201(input_tensor=input_layer, include_top=False, weights=weights)
    base.trainable = True
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_class)(x)
    preds = layers.Activation("softmax", dtype=tf.float32)(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=preds)
    return model


def mobilenet(img_size=(224,224), num_class=2, weights="imagenet", dtype=tf.float32):
    input_layer = layers.Input(shape=(img_size[0],img_size[1],3), dtype=dtype)
    base = models.MobileNetV2(input_tensor=input_layer, include_top=False, weights=weights)
    base.trainable = True
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_class)(x)
    preds = layers.Activation("softmax", dtype=tf.float32)(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=preds)
    return model


def convert_for_training(model, wd=0.0001, verbose=False):
    model_config = model.get_config()
    for layer, layer_config in zip(model.layers, model_config["layers"]):
        if hasattr(layer, "kernel_regularizer"):
            if verbose:
                print("Adjust kernel_regularizer for", layer.name)
            regularizer = tf.keras.regularizers.l2(wd)
            layer_config["config"]["kernel_regularizer"] = {
                "class_name": regularizer.__class__.__name__,
                "config": regularizer.get_config()
            }
        if str(type(layer)) == "<class 'tensorflow.python.keras.layers.normalization.BatchNormalization'>":
            if verbose:
                print("Adjust BatchNorm settings for", layer.name)
            layer_config["config"]["momentum"] = 0.9
            layer_config["config"]["epsilon"] = 1e-5
            layer_config["config"]["fused"] = True
    del model
    model = tf.keras.models.Model.from_config(model_config)
    model.trainable = True
    return model

