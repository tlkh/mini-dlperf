import tensorflow as tf

class DecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Piecewise constant decay with warmup schedule.
    """

    def __init__(self, base_lr, warmup_steps, max_steps, min_lr=1e-4, name="PiecewiseConstantDecayWithWarmup"):
        super(DecayWithWarmup, self).__init__()
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_steps = max_steps + 1
        self.warmup_steps = warmup_steps
        self.name = name

    def __call__(self, step):
        with tf.device("/CPU:0"):
            warmup_lr = self.base_lr * step/self.warmup_steps
            main_lr = self.base_lr * ((self.max_steps-step)/self.max_steps)**(10*step/self.max_steps)
            return tf.math.minimum(warmup_lr, main_lr) + self.min_lr
