import time
import tensorflow as tf


class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TimeHistory, self).__init__()
        self.epoch_times = None
    def on_train_begin(self, logs={}):
        self.epoch_times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        epoch_duration = time.time() - self.epoch_time_start
        self.epoch_times.append(epoch_duration)
        
        
        