import time
import tensorflow.compat.v2 as tf


class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, img_per_epoch):
        self.times = None
        self.img_per_epoch = img_per_epoch
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        epoch_duration = time.time() - self.epoch_time_start
        self.times.append(epoch_duration)
        img_per_sec = int(self.img_per_epoch/epoch_duration)
        print("\nImage/sec:", img_per_sec)
