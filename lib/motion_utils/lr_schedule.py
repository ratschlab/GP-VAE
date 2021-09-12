import tensorflow as tf
# scheduled learning rate with weight decay and warmup steps.
# as in the ULMFIT paper. scheduler code from:
# https://hwiyong.tistory.com/421

def get_lr(g_step, warmup_step=5000, init_lr=1e-1, min_lr=1e-5, power=1.):
    lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=g_step - warmup_step,
        end_learning_rate=min_lr,
        power=power
    )
    lr = LRSchedule(init_lr, warmup_step, lr_scheduler)
    return lr


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, warmup_step, decay_fn):
        self.init_lr = init_lr
        self.warmup_step = warmup_step
        self.decay_fn = decay_fn

    def __call__(self, step):
        if step == 0:
            step += 1

        step_float = tf.cast(step, tf.float32)
        warmup_step_float = tf.cast(self.warmup_step, tf.float32)

        cur_lr = tf.cond(step_float < warmup_step_float, lambda: self.init_lr * (step_float / warmup_step_float),
                       lambda: self.decay_fn(step_float - warmup_step_float), )
        return cur_lr


if __name__ == '__main__':
    # example use
    data_size = 100000
    batch_size = 512
    global_step = data_size // batch_size
    warmup_step = int(global_step * 0.6)
    init_lr = 0.1
    min_lr = 1e-6
    power = 1.

    lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=global_step - warmup_step,
        end_learning_rate=min_lr,
        power=power
    )

    lr_schedule = LRSchedule(init_lr, warmup_step, lr_scheduler)