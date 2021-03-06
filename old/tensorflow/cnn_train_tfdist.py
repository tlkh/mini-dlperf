import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rn152", action="store_true", default=False,
                    help="Train a larger ResNet-152 model instead of ResNet-50")
parser.add_argument("--rn50v2", action="store_true", default=False,
                    help="Train ResNet-50 V2 model instead of ResNet-50 MLPerf")
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
parser.add_argument("--warmup_epochs", default=5, type=int)
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
parser.add_argument("--report", action="store_true", default=False)
args = parser.parse_args()

import os
import multiprocessing
n_cores = multiprocessing.cpu_count()
print("Number of logical cores:", n_cores)
worker_threads = n_cores
print("Number of threads used for dataloader:", worker_threads)
os.environ["TF_DISABLE_NVTX_RANGES"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = str(n_cores)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
import logging
logger = logging.getLogger()
logger.setLevel(logging.WARN)
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from common import dataloaders, cnn_models, callbacks, schedules, ops
if args.stats:
    from nvstatsrecorder.callbacks import NVStats, NVLinkStats

print("Using XLA:", args.xla)
tf.config.optimizer.set_jit(args.xla)
print("Using grappler AMP:", args.amp)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.amp,
                                              "debug_stripper": True})
tf.config.threading.set_inter_op_parallelism_threads(n_cores)

strategy = tf.distribute.MirroredStrategy()
replicas = strategy.num_replicas_in_sync

BATCH_SIZE = args.batch_size * replicas
IMG_SIZE = args.img_size
L_IMG_SIZE = int(args.img_size*1.1)
EPOCHS = args.epochs
tf_image_dtype = tf.float16

print("Number of devices:", replicas)
print("Global batch size:", BATCH_SIZE)
print("Base learning rate:", args.lr)

print("Loading Dataset")

print("Using TFDS dataset:", args.dataset)
    
dataset = dataloaders.return_fast_tfds(args.dataset,
                                       data_dir=args.data_dir,
                                       worker_threads=worker_threads,
                                       buffer=BATCH_SIZE*2)

num_class = dataset["num_class"]

PAD = False
if PAD:
    if num_class == 10:
        num_class = 16
    elif num_class == 1000:
        num_class = 1024
    print("Padded final layer to", num_class)
    
num_train = dataset["num_train"]
num_valid = dataset["num_valid"]

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

means = tf.cast(tf.broadcast_to(CHANNEL_MEANS, (IMG_SIZE, IMG_SIZE, 3)), tf_image_dtype)

if args.img_aug:
    @tf.function
    def format_train_example(image_path, label):
        image = tf.io.decode_jpeg(image_path, channels=3, ratio=2,
                              fancy_upscaling=False,
                              dct_method="INTEGER_FAST")
        image = ops.resize_preserve_ratio(image, L_IMG_SIZE)
        image = ops.augment_image(image, IMG_SIZE)
        image = (tf.cast(image, tf_image_dtype) - means)/127.5
        return image, label
else:
    @tf.function
    def format_train_example(image_path, label):
        image = tf.io.decode_jpeg(image_path, channels=3, ratio=2,
                                  fancy_upscaling=False,
                                  dct_method="INTEGER_FAST")
        image = tf.image.central_crop(image, 0.9)
        image = ops.crop_center_and_resize(image, IMG_SIZE)
        image = (tf.cast(image, tf_image_dtype) - means)/127.5
        return image, label


@tf.function
def format_test_example(image_path, label):
    image = tf.io.decode_jpeg(image_path, channels=3, ratio=2,
                              fancy_upscaling=False,
                              dct_method="INTEGER_FAST")
    image = tf.image.central_crop(image, 0.9)
    image = ops.crop_center_and_resize(image, IMG_SIZE)
    image = (tf.cast(image, tf_image_dtype) - means)/127.5
    return image, label

print("Build tf.data input pipeline")

train = dataset["train"]
train = train.map(format_train_example, num_parallel_calls=multiprocessing.cpu_count())
train = train.batch(BATCH_SIZE, drop_remainder=True)
train = train.prefetch(1)

valid = dataset["valid"]
valid = valid.map(format_test_example, num_parallel_calls=multiprocessing.cpu_count())
if num_valid > 512 :
    VAL_BATCH_SIZE = BATCH_SIZE
else:
    VAL_BATCH_SIZE = num_valid//2
valid = valid.batch(VAL_BATCH_SIZE, drop_remainder=False)
valid = valid.prefetch(1)

train_steps = int(num_train/BATCH_SIZE)
valid_steps = int(num_valid/VAL_BATCH_SIZE)

print("Running pipelines:")

for batch in train.take(2):
    image, label = batch[0].numpy(), batch[1].numpy()
    print("* Image shape:", image.shape)
    print("* Image size:", len(str(image)))
    print("* Label shape:", label.shape)

time.sleep(1)
    
for batch in valid.take(2):
    image, label = batch[0].numpy(), batch[1].numpy()
    print("* Image shape:", image.shape)
    print("* Image size:", len(str(image)))
    print("* Label shape:", label.shape)
    
time.sleep(1)

print("Build and distribute model")

if args.keras_amp:
    print("Using Keras AMP:", args.keras_amp)
    tf.keras.mixed_precision.experimental.set_policy("mixed_float16")
    
with strategy.scope():
    if args.rn152:
        print("Using ResNet-152 V2 model")
        model = cnn_models.rn152((IMG_SIZE,IMG_SIZE), num_class, weights=None, dtype=tf_image_dtype)
        model = cnn_models.convert_for_training(model)
    elif args.dn201:
        print("Using DenseNet-201 model")
        model = cnn_models.dn201((IMG_SIZE,IMG_SIZE), num_class, weights=None, dtype=tf_image_dtype)
        model = cnn_models.convert_for_training(model)
    elif args.mobilenet:
        print("Using MobileNetV2 model")
        model = cnn_models.mobilenet((IMG_SIZE,IMG_SIZE), num_class, weights=None, dtype=tf_image_dtype)
        model = cnn_models.convert_for_training(model)
    elif args.rn50v2:
        print("Using ResNet-50 V2 model")
        model = cnn_models.rn50((IMG_SIZE,IMG_SIZE), num_class, weights=None, dtype=tf_image_dtype)
        model = cnn_models.convert_for_training(model)
    else:
        print("Using ResNet-50 MLPerf model")
        model = cnn_models.rn50_mlperf((IMG_SIZE,IMG_SIZE), num_class)
    
    schedule = schedules.WarmupExpDecay(
        epoch_steps=train_steps,
        base_lr=args.lr,
        min_lr=0.01*args.lr,
        decay_exp=6,
        warmup_epochs=args.warmup_epochs,
        flat_epochs=30,
        max_epochs=EPOCHS,
    )
    
    opt = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.875, nesterov=True)
    
    if args.amp:
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
        
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    top_1 = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
    top_5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_acc")
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[top_1, top_5])
    print("Model metrics:", model.metrics_names)

print("Train model")

if args.steps:
    train_steps = args.steps
    if valid_steps > args.steps:
        valid_steps = args.steps

verbose = args.verbose
if verbose != 1:
    print("Verbose level:", verbose)
    print("You will not see progress during training!")
    
train_img_per_epoch = train_steps * BATCH_SIZE

time_callback = callbacks.TimeHistory(eg_per_epoch=train_img_per_epoch)
checkpoint_name = str(int(time.time())) + "_checkpoint.h5"
checkpoints = tf.keras.callbacks.ModelCheckpoint(checkpoint_name, monitor="val_acc",
                                                 verbose=1, save_best_only=True, save_weights_only=True)

callbacks = [time_callback, checkpoints]

if args.stats:
    try:
        SUDO_PASSWORD = os.environ["SUDO_PASSWORD"]
        nv_stats = NVStats(gpu_index=0, interval=5, tensor_util=True, sudo_password=SUDO_PASSWORD)
        nvlink_stats = NVLinkStats(SUDO_PASSWORD, gpus=[0,1,2,3], interval=5)
        callbacks.append(nvlink_stats)
        RECORD_NVLINK = True
    except Exception as e:
        print(e)
        print("No sudo access, not recording Tensor Core and NVLink utilization")
        nv_stats = NVStats(gpu_index=0, interval=5, tensor_util=False)
        RECORD_NVLINK = False
    callbacks.append(nv_stats)
    

if train_steps < 20:
    validation_freq = 2
else:
    validation_freq = 1
    
print("Start training")

train_start = time.time()

if args.no_val:
    with strategy.scope():
        model.fit(train, steps_per_epoch=train_steps,
                  epochs=EPOCHS, callbacks=callbacks, verbose=verbose)
else:
    with strategy.scope():
        model.fit(train, steps_per_epoch=train_steps, validation_freq=1, 
                  validation_data=valid, validation_steps=valid_steps,
                  epochs=EPOCHS, callbacks=callbacks, verbose=verbose) 
    
train_end = time.time()

if args.stats:
    nv_stats_recorder = nv_stats.recorder
    prefix = args.dataset.replace("/", "_")
    nv_stats_recorder.plot_gpu_util(smooth=5, outpath=prefix+"_resnet_gpu_util.jpg")
    nv_stats_recorder.summary()
    if RECORD_NVLINK:
        nvlink_stats_recorder = nvlink_stats.recorder
        nvlink_stats_recorder.plot_nvlink_traffic(smooth=5, outpath=prefix+"_resnet_nvlink_util.jpg")

duration = min(time_callback.times)
fps = train_steps*BATCH_SIZE/duration

try:
    print("Loading best checkpoint")
    model.load_weights(checkpoint_name)
except Exception as e:
    print(e)
    print("Not loading any checkpoint")

with strategy.scope():
    loss, acc, top_5_acc = model.evaluate(valid, steps=valid_steps)

print("\n")
print("Results:")
print("========\n")
print("ResNet FPS:")
print("*", replicas, "GPU:", int(fps))
print("* Per GPU:", int(fps/replicas))
print("Total train time:", int(train_end-train_start))
print("Acc:", acc, "Top 5 Acc:", top_5_acc)

if args.report:
    import paramiko
    
    uid = str(int(time.time()))
    result_path = "/home/jovyan/"+uid+"_result.txt"
    f = open(result_path,"w+")
    f.write("Avg images/sec:"+str(fps))
    f.write("\nEnd-to-end time:"+str(train_end-train_start))
    f.write("\n")
    f.close()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("192.168.33.10", username="uat", password=os.environ["uat_password"])
    sftp = ssh.open_sftp()
    sftp.put(result_path, "/home/users/uat/"+uid+"_result.txt")
    sftp.close()
    ssh.close()
