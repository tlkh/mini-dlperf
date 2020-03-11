import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rn152", action="store_true", default=False,
                    help="Train a larger ResNet-152 model instead of ResNet-50")
parser.add_argument("--dn201", action="store_true", default=False,
                    help="Train a larger DenseNet-201 model instead of ResNet-50")
parser.add_argument("--mobilenet", action="store_true", default=False,
                    help="Train a smaller MobileNetV2 model instead of ResNet-50")
parser.add_argument("--amp", action="store_true", default=False,
                    help="Use grappler AMP for mixed precision training")
parser.add_argument("--keras_amp", action="store_true", default=False,
                    help="Use Keras AMP for mixed precision training")
parser.add_argument("--xla", action="store_true", default=False,
                    help="Use XLA compiler")
parser.add_argument("--batch_size", default=128, type=int,
                    help="Batch size to use for training")
parser.add_argument("--img_size", default=224, type=int,
                    help="Image size to use for training")
parser.add_argument("--lr", default=0.1, type=float,
                    help="Learning rate")
parser.add_argument("--epochs", default=90, type=int,
                    help="Number of epochs to train for")
parser.add_argument("--stats", action="store_true", default=False,
                    help="Record stats using NVStatsRecorder")
parser.add_argument("--dataset", default="imagenette/160px",
                    help="TFDS Dataset to train on")
parser.add_argument("--data_dir", default="~/tensorflow_datasets",
                    help="TFDS Dataset directory")
parser.add_argument("--verbose", default=1, type=int)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--no_val", action="store_true", default=False)
parser.add_argument("--img_aug", action="store_true", default=False)
args = parser.parse_args()

import os
import psutil
psutil.cpu_percent(interval=None)
import multiprocessing
worker_threads = multiprocessing.cpu_count()
os.environ["TF_DISABLE_NVTX_RANGES"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = str(worker_threads)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
import logging
logger = logging.getLogger()
logger.setLevel(logging.WARN)
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from common import dataloaders, cnn_models, callbacks, schedules

if args.stats:
    from nvstatsrecorder.callbacks import NVStats, NVLinkStats

print("Using XLA:", args.xla)
tf.config.optimizer.set_jit(args.xla)
print("Using grappler AMP:", args.amp)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.amp})
tf.config.threading.set_inter_op_parallelism_threads(worker_threads)

strategy = tf.distribute.MirroredStrategy()
replicas = strategy.num_replicas_in_sync

BATCH_SIZE = args.batch_size * replicas
IMG_SIZE = args.img_size
IMG_SIZE_C = (args.img_size, args.img_size, 3)
L_IMG_SIZE = (int(args.img_size*1.1), int(args.img_size*1.1))
EPOCHS = args.epochs

print("Number of devices:", replicas)
print("Global batch size:", BATCH_SIZE)
print("Base learning rate:", args.lr)

print("Loading Dataset")

print("Using TFDS dataset:", args.dataset)
    
dataset = dataloaders.return_fast_tfds(args.dataset,
                                       data_dir=args.data_dir,
                                       worker_threads=worker_threads,
                                       buffer=16384)

num_class = dataset["num_class"]
num_train = dataset["num_train"]
num_valid = dataset["num_valid"]

if args.img_aug:
    
    @tf.function
    def format_train_example(_image, label):
        image = tf.io.decode_jpeg(_image, channels=3,
                                  fancy_upscaling=False,
                                  dct_method="INTEGER_FAST")
        #image = tf.image.resize_with_pad(image, L_IMG_SIZE, L_IMG_SIZE)
        image = tf.image.resize(image, L_IMG_SIZE)
        image = tf.image.random_crop(image, IMG_SIZE_C)
        image = tf.image.random_brightness(image, max_delta=32/255)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_flip_left_right(image)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.clip_by_value(image, 0, 1.0)
        label = tf.one_hot(label, num_class)
        return image, label
else:
    @tf.function
    def format_train_example(_image, label):
        image = tf.io.decode_jpeg(_image, channels=3,
                                  fancy_upscaling=False,
                                  dct_method="INTEGER_FAST")
        #image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, num_class)
        return image, label


@tf.function
def format_test_example(_image, label):
    image = tf.io.decode_jpeg(_image, channels=3,
                              fancy_upscaling=False,
                              dct_method="INTEGER_FAST")
    image = tf.image.central_crop(image, 0.9)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, num_class)
    return image, label

print("Build tf.data input pipeline")

train = dataset["train"]
train = train.map(format_train_example, num_parallel_calls=worker_threads)
train = train.batch(BATCH_SIZE, drop_remainder=True)
train = train.prefetch(16)

valid = dataset["valid"]
valid = valid.map(format_test_example, num_parallel_calls=worker_threads)
if num_valid > 512 :
    VAL_BATCH_SIZE = BATCH_SIZE
else:
    VAL_BATCH_SIZE = num_valid//replicas
valid = valid.batch(VAL_BATCH_SIZE, drop_remainder=False)
valid = valid.prefetch(16)

train_steps = int(num_train/BATCH_SIZE)
valid_steps = int(num_valid/VAL_BATCH_SIZE)

print("Running pipelines:")

for batch in train.take(2):
    image, label = batch[0].numpy(), batch[1].numpy()
    print("* Image shape:", image.shape)
    print("* Image size:", len(str(image)))
    print("* Label shape:", label.shape)

for batch in valid.take(2):
    image, label = batch[0].numpy(), batch[1].numpy()
    print("* Image shape:", image.shape)
    print("* Image size:", len(str(image)))
    print("* Label shape:", label.shape)
    
print("Wait for built prefetch cache")
while psutil.cpu_percent(interval=1.0) > 1/worker_threads*100:
    time.sleep(1)
print("CPU:", psutil.cpu_percent(interval=None))
print("Done!")

print("Build and distribute model")

if args.keras_amp:
    print("Using Keras AMP:", args.keras_amp)
    tf.keras.mixed_precision.experimental.set_policy("mixed_float16")
    
with strategy.scope():
    if args.rn152:
        print("Using ResNet-152 model")
        model = cnn_models.rn152((IMG_SIZE,IMG_SIZE), num_class, weights=None)
    elif args.dn201:
        print("Using DenseNet-201 model")
        model = cnn_models.dn201((IMG_SIZE,IMG_SIZE), num_class, weights=None)
    elif args.mobilenet:
        print("Using MobileNetV2 model")
        model = cnn_models.mobilenet((IMG_SIZE,IMG_SIZE), num_class, weights=None)
    else:
        print("Using ResNet-50 model")
        model = cnn_models.rn50((IMG_SIZE,IMG_SIZE), num_class, weights=None)
        
    model = cnn_models.convert_for_training(model)
    
    warmup_epochs = 5

    schedule = schedules.DecayWithWarmup(
        epoch_steps=train_steps,
        base_lr=args.lr,
        min_lr=0.001,
        decay_exp=6,
        warmup_epochs=warmup_epochs,
        flat_epochs=30,
        max_epochs=EPOCHS,
    )
    
    opt = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.9)
    #opt = tf.keras.optimizers.Adam(learning_rate=schedule)
    
    if args.amp:
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["acc"])
    """
    try:
        model.load_weights("checkpoint.h5")
        print("Loaded weights from checkpoint")
    except Exception as e:
        print(e)
        print("Not resuming from checkpoint")
    """

print("Train model")

verbose = args.verbose
if verbose != 1:
    print("Verbose level:", verbose)
    print("You will not see progress during training!")
time_callback = callbacks.TimeHistory(img_per_epoch=train_steps*BATCH_SIZE)
checkpoints = tf.keras.callbacks.ModelCheckpoint("checkpoint.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)

callbacks = [time_callback, checkpoints]

if args.stats:
    SUDO_PASSWORD = os.environ["SUDO_PASSWORD"]
    nv_stats = NVStats(gpu_index=0, interval=5, tensor_util=True, sudo_password=SUDO_PASSWORD)
    nvlink_stats = NVLinkStats(SUDO_PASSWORD, gpus=[0,1,2,3], interval=5)
    callbacks.append(nv_stats)
    callbacks.append(nvlink_stats)

if args.steps:
    train_steps = args.steps
    if valid_steps > args.steps:
        valid_steps = args.steps
    
print("Start training")

train_start = time.time()

if args.no_val:
    with strategy.scope():
        model.fit(train, steps_per_epoch=train_steps,
                  epochs=EPOCHS, callbacks=callbacks, verbose=verbose)
else:
    with strategy.scope():
        model.fit(train, steps_per_epoch=train_steps, validation_freq=2, 
                  validation_data=valid, validation_steps=valid_steps,
                  epochs=EPOCHS, callbacks=callbacks, verbose=verbose) 
    
train_end = time.time()

if args.stats:
    nv_stats_recorder = nv_stats.recorder
    nvlink_stats_recorder = nvlink_stats.recorder
    prefix = args.dataset.replace("/", "_")
    nv_stats_recorder.plot_gpu_util(smooth=5, outpath=prefix+"_resnet_gpu_util.png")
    nvlink_stats_recorder.plot_nvlink_traffic(smooth=5, outpath=prefix+"_resnet_nvlink_util.png")

duration = min(time_callback.times)
fps = train_steps*BATCH_SIZE/duration

try:
    print("Loading best checkpoint")
    model.load_weights("checkpoint.h5")
except Exception as e:
    print(e)
    print("Not loading any checkpoint")

with strategy.scope():
    loss, acc = model.evaluate(valid, steps=valid_steps)

print("\n")
print("Results:")
print("========\n")
print("ResNet FPS:")
print("*", replicas, "GPU:", int(fps))
print("* Per GPU:", int(fps/replicas))
print("Total train time:", int(train_end-train_start))
print("Loss:", loss)
print("Acc:", acc)
