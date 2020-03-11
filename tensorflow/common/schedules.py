import tensorflow as tf

class DecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, base_lr, warmup_epochs, flat_epochs, max_epochs, epoch_steps, min_lr=1e-4, decay_exp=5, name="DecayWithWarmup"):
        super(DecayWithWarmup, self).__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_epochs * epoch_steps
        self.flat_steps = flat_epochs * epoch_steps
        self.max_steps = (max_epochs + 1) * epoch_steps
        self.min_lr = min_lr
        self.decay_exp = decay_exp
        self.name = name

    def __call__(self, step):
        with tf.device("/CPU:0"):
            warmup_lr = self.base_lr * step/self.warmup_steps
            flat_lr = self.base_lr
            lr = tf.math.minimum(warmup_lr, flat_lr)
            decay_lr = self.base_lr * ((self.max_steps-step+self.flat_steps)/self.max_steps)**self.decay_exp
            lr = tf.math.minimum(lr, decay_lr)
            return lr + self.min_lr
