import time
import tensorflow as tf


class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, eg_per_epoch):
        super(TimeHistory, self).__init__()
        self.times = None
        self.eg_per_epoch = eg_per_epoch
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        epoch_duration = time.time() - self.epoch_time_start
        self.times.append(epoch_duration)
        print("\nEg/sec:", int(self.eg_per_epoch/epoch_duration))

        
class LRWarmUp(tf.keras.callbacks.Callback):
    def __init__(self, warmup_steps, min_lr=1e-4):
        super(LRWarmUp, self).__init__()
        self.global_step = 0
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.target_lr = None
    def _get_current_lr(self):
        return float(tf.keras.backend.get_value(self.model.optimizer.lr))
    def _get_step_lr(self):
        return (self.target_lr * self.global_step / self.warmup_steps) + self.min_lr
    def on_train_begin(self, logs={}):
        self.target_lr = self._get_current_lr()
        print("Begin learning rate warm-up to", self._get_current_lr())
    def on_train_batch_begin(self, batch, logs={}):
        if self.global_step < self.warmup_steps:
            self.global_step += 1
            tf.keras.backend.set_value(self.model.optimizer.lr, self._get_step_lr())
    def on_epoch_begin(self, epoch, logs={}):
        tf.keras.backend.set_value(self.model.optimizer.lr, self._get_step_lr())
        if self.global_step < self.warmup_steps:
            print("\nEpoch %05d start: Learning rate is %6.4f." % (epoch, self._get_current_lr()))
    def on_epoch_end(self, epoch, logs={}):
        if self.global_step < self.warmup_steps:
            print("\nEpoch %05d end: Learning rate is %6.4f." % (epoch, self._get_current_lr()))
        