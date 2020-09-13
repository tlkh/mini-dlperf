import time
import os
import multiprocessing
os.environ["TF_DISABLE_NVTX_RANGES"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
import horovod.tensorflow.keras as hvd
import argparse

parser = argparse.ArgumentParser(
    description="Train and evaluate Transformers for various GLUE tasks",
)

parser.add_argument("--amp", action="store_true", default=True,
                    help="Use grappler AMP for mixed precision training")
parser.add_argument("--xla", action="store_true", default=True,
                    help="Use XLA compiler")
parser.add_argument("--fp16comp", action="store_true", default=True,
                    help="Use float16 compression during allreduce")
parser.add_argument("--epochs", default=3,
                    help="Number of epochs to train for",
                    type=int)
parser.add_argument("--interval", default=1,
                    help="Number of fake epochs per real epoch",
                    type=int)
parser.add_argument("--batch_size", default=8,
                    help="Batch size to use for training",
                    type=int)
parser.add_argument("--lr", default=1e-5,
                    help="Learning Rate to use for training",
                    type=float)
parser.add_argument("--warmup_prop", default=1.0,
                    help="Proportion of steps to use for LR warmup",
                    type=float)
parser.add_argument("--maxseqlen", default=128,
                    help="Maximum input sequence length",
                    type=int)
parser.add_argument("--task", default="mrpc",
                    help="Task for training and evaluation")
parser.add_argument("--model", default="bert-large-cased-whole-word-masking",
                    help="Which Transformer model to use")
parser.add_argument("--stats", action="store_true", default=False,
                    help="Record stats using NVStatsRecorder")
args = parser.parse_args()

LEARNING_RATE = args.lr
USE_XLA = args.xla
USE_AMP = args.amp
model_name = args.model
MAX_SEQ_LEN = args.maxseqlen
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

hvd.init()
hvd_rank = hvd.local_rank()
hvd_size = hvd.size()

worker_threads = multiprocessing.cpu_count()//hvd_size

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(physical_devices[hvd_rank], 'GPU')

import transformers as xfmers
from common import dataloaders, xfmer_models, callbacks, schedules
if args.stats:
    from nvstatsrecorder.callbacks import NVStats, NVLinkStats

# training parameters

task_list = {
    "cola": "glue/cola",
    "mrpc": "glue/mrpc",
    "sst-2": "glue/sst2",
    "qqp": "glue/qqp",
    "mnli": "glue/mnli",
    "qnli": "glue/qnli"
}

task_name = args.task
if task_name == "sst2":
    task_name = "sst-2"
dataset_name = task_list[task_name]

tf.config.threading.set_inter_op_parallelism_threads(worker_threads)
tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({
    "auto_mixed_precision": USE_AMP,
    "debug_stripper": True,
})

# Building input pipeline

tokenizer = dataloaders.create_tokenizer(model_name)
task_dataset = dataloaders.return_glue_task(tokenizer, dataset_name, task_name, MAX_SEQ_LEN,
                                            index=hvd_rank, num_shards=hvd_size)
train_dataset = task_dataset["train_dataset"]
train_dataset = train_dataset.repeat().shuffle(task_dataset["train_examples"])
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(64)

valid_dataset = task_dataset["valid_dataset"].repeat()
val_batchsize = BATCH_SIZE*2
valid_dataset = valid_dataset.batch(val_batchsize, drop_remainder=False).prefetch(64)

test_dataset = task_dataset["test_dataset"].repeat()
test_dataset = test_dataset.batch(val_batchsize, drop_remainder=False).prefetch(64)

print("Running pipelines:")

for batch in train_dataset.take(1):
    batch = str(batch)
    print("Data length:", len(batch))

for batch in test_dataset.take(1):
    batch = str(batch)
    print("Data length:", len(batch))
    
for batch in valid_dataset.take(1):
    batch = str(batch)
    print("Data length:", len(batch))
    
time.sleep(1)
    
print(hvd_rank, "Building model...")

model = xfmer_models.create_model(model_name, task_dataset["num_labels"])
opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-08)
if args.fp16comp:
    print("Using float16 compression for all-reduce")
    compression = hvd.Compression.fp16
else:
    compression = hvd.Compression.none
opt = hvd.DistributedOptimizer(opt,
                               compression=compression,
                               sparse_as_dense=True)
if USE_AMP:
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=opt,
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy("accuracy")],
              experimental_run_tf_function=False)

# train model

fake_epochs_ratio = int(args.interval)
    
train_steps_per_epoch = int(task_dataset["train_examples"]/BATCH_SIZE/hvd_size) -1
valid_steps_per_epoch = int(task_dataset["valid_examples"]/val_batchsize/hvd_size) - 1

warmup_epochs = int(EPOCHS*fake_epochs_ratio*args.warmup_prop)

print("Warmup epochs:", warmup_epochs)

lr_schedule = hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, verbose=1,
                                                       steps_per_epoch=int(train_steps_per_epoch/fake_epochs_ratio))
hvd_broadcast = hvd.callbacks.BroadcastGlobalVariablesCallback(0)
hvd_metric = hvd.callbacks.MetricAverageCallback()

callbacks_list = [hvd_broadcast, hvd_metric, lr_schedule]

if hvd_rank == 0:
    if USE_XLA:
        print("XLA is enabled. First run will be delayed due to XLA JIT compilation.")
    if USE_AMP:
        print("Model is using Automatic Mixed Precision")
    verbose = 1
    model.summary()
    time_callback = callbacks.TimeHistory(eg_per_epoch=BATCH_SIZE*hvd_size*int(train_steps_per_epoch/fake_epochs_ratio))
    checkpoint_name = str(int(time.time()))+"_checkpoint.h5"
    checkpoints = tf.keras.callbacks.ModelCheckpoint(monitor="val_accuracy", mode="max",
                                                     filepath=checkpoint_name,
                                                     save_weights_only=True,
                                                     save_best_only=True)
    callbacks_list.append(time_callback)
    callbacks_list.append(checkpoints)
    if args.stats:
        try:
            SUDO_PASSWORD = os.environ["SUDO_PASSWORD"]
            nv_stats = NVStats(gpu_index=0, interval=5, tensor_util=True, sudo_password=SUDO_PASSWORD)
            nvlink_stats = NVLinkStats(SUDO_PASSWORD, gpus=[0,1,2,3], interval=5)
            callbacks_list.append(nv_stats)
            callbacks_list.append(nvlink_stats)
        except Exception as e:
            print("No sudo access, not recording Tensor Core or NVLink metrics")
            nv_stats = NVStats(gpu_index=0, interval=5, tensor_util=False)
            callbacks_list.append(nv_stats)
            NVLINK_STATS = False
else:
    verbose = 0
    
print(hvd_rank, "Starting training...")
print(hvd_rank, "Steps:", train_steps_per_epoch, valid_steps_per_epoch)

if hvd_rank == 0:
    time.sleep(hvd_size)
    script_start_time = time.time()

log = model.fit(train_dataset,
                epochs=EPOCHS*fake_epochs_ratio, steps_per_epoch=int(train_steps_per_epoch/fake_epochs_ratio),
                validation_data=valid_dataset, validation_steps=valid_steps_per_epoch,
                validation_freq=2, callbacks=callbacks_list, verbose=verbose)

if hvd_rank == 0:
    script_end_time = time.time()
    model.load_weights(checkpoint_name)
    score = model.evaluate(test_dataset, steps=int(task_dataset["test_examples"]/val_batchsize))
    
    if args.stats:
        SMOOTH = 10
        nv_stats_recorder = nv_stats.recorder
        if NVLINK_STATS:
            nvlink_stats_recorder = nvlink_stats.recorder
            nvlink_stats_recorder.plot_nvlink_traffic(smooth=SMOOTH, outpath="transformer_nvlink_util.jpg")
        nv_stats_recorder.plot_gpu_util(smooth=SMOOTH, outpath="transformer_gpu_util.jpg")
        nv_stats_recorder.plot_gpu_temp(smooth=SMOOTH, outpath="transformer_gpu_temp.jpg")
        nv_stats_recorder.summary()

    # results
    
    total_time = int(script_end_time - script_start_time)
    cold_start_duration = max(time_callback.times)
    epoch_duration = min(time_callback.times)
    eg_per_sec = int(train_steps_per_epoch/fake_epochs_ratio*BATCH_SIZE/epoch_duration)

    print("\n\n=================\n\n")
    print("\nResults (DIST/XLA/AMP:", "horovod", USE_XLA, USE_AMP, ")", task_name)
    print("Total time:", total_time, "seconds")
    print("Cold Start time:", int(cold_start_duration - epoch_duration))
    print("Training Throughput:", eg_per_sec * hvd_size, "examples per second")

    print("Throughput per GPU:", eg_per_sec, "examples per second")
    
    print("Loss:", score[0])
    print("Accuracy:", round(score[1],4))
    print("\n\n=================\n\n")
