import argparse

parser = argparse.ArgumentParser(
    description="Train and evaluate T5 model",
)

parser.add_argument("--amp", action="store_true", default=True,
                    help="Use grappler AMP for mixed precision training")
parser.add_argument("--xla", action="store_true", default=False,
                    help="Use XLA compiler")
parser.add_argument("--fp16comp", action="store_true", default=True,
                    help="Use float16 compression during allreduce")
parser.add_argument("--reduce_vram", action="store_true", default=False,
                    help="Optimize VRAM usage for large models")
parser.add_argument("--epochs", default=4,
                    help="Number of epochs to train for",
                    type=int)
parser.add_argument("--batch_size", default=8,
                    help="Batch size to use for training",
                    type=int)
parser.add_argument("--steps", default=200,
                    help="Number of steps to use for training",
                    type=int)
parser.add_argument("--maxseqlen", default=512,
                    help="Maximum input sequence length",
                    type=int)
parser.add_argument("--model", default="t5-small",
                    help="Which T5 model to use")
args = parser.parse_args()

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=fusible"

if args.xla:
    if os.environ.get("TF_XLA_FLAGS", None) is not None:
        os.environ["TF_XLA_FLAGS"] += " --tf_xla_enable_lazy_compilation false"
    else:
        os.environ["TF_XLA_FLAGS"] = " --tf_xla_enable_lazy_compilation false"
    os.environ["TF_XLA_FLAGS"] += " --tf_xla_async_io_level 1"

import tensorflow as tf
tf.config.optimizer.set_jit(args.xla)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.amp,
                                              "debug_stripper": True})
import numpy as np
import transformers
from common import callbacks
import horovod.tensorflow.keras as hvd
from nvstatsrecorder.callbacks import NVStats, NVLinkStats

hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# synthetic data

N = 1024
L = args.maxseqlen

inputs, targets = np.ones((N, L), dtype="int"), np.ones((N, L), dtype="int")


dataset = tf.data.Dataset.from_tensor_slices(({"input_1": inputs,
                                               "input_2": inputs,
                                               "input_3": inputs,
                                               "input_4": inputs}, targets))
dataset = dataset.repeat().batch(args.batch_size).prefetch(8)

def return_t5_model(model_name, max_seq_len=512):
    config = transformers.T5Config.from_pretrained(model_name)
    xfmer = transformers.TFT5Model.from_pretrained(model_name, config=config)
    xfmer.trainable = True
    
    l_input_sentence = tf.keras.layers.Input(shape=[max_seq_len,], dtype=tf.int32)
    l_input_attn_mask = tf.keras.layers.Input(shape=[max_seq_len,], dtype=tf.int32)
    l_decoder_input_ids = tf.keras.layers.Input(shape=[max_seq_len,], dtype=tf.int32)
    l_decoder_attention_mask = tf.keras.layers.Input(shape=[max_seq_len,], dtype=tf.int32)
    
    preds = xfmer({"input_ids": l_input_sentence,
                     "attention_mask": l_input_attn_mask,
                     "decoder_input_ids": l_decoder_input_ids,
                     "decoder_attention_mask": l_decoder_attention_mask})[0]

    model = tf.keras.models.Model(inputs=[l_input_sentence, l_input_attn_mask, l_decoder_input_ids, l_decoder_attention_mask],
                                  outputs=preds)
    return model



model = return_t5_model(model_name=args.model, max_seq_len=args.maxseqlen)
opt = tf.keras.optimizers.Adam()

if args.fp16comp:
    print("Using float16 compression for all-reduce")
    compression = hvd.Compression.fp16
else:
    compression = hvd.Compression.none
    
if args.reduce_vram:
    opt = hvd.DistributedOptimizer(opt,
                                   sparse_as_dense=False,
                                   device_sparse='/cpu:0',
                                   compression=compression)
else:
    opt = hvd.DistributedOptimizer(opt,
                                   sparse_as_dense=True,
                                   compression=compression)

if args.amp:
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")

lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(loss=lossfn,
              optimizer=opt,
              experimental_run_tf_function=False)

time_history = callbacks.TimeHistory()

callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    time_history,
]

if hvd.local_rank() == 0:
    nv_stats = NVStats(gpu_index=0, interval=1, verbose=False)
    nvlink_interval = 5
    nvlink_stats = NVLinkStats(None, gpus=[0,1,2,3], interval=nvlink_interval)
    callbacks.append(nv_stats)
    callbacks.append(nvlink_stats)
    RECORD_NVLINK = True

steps_per_epoch = args.steps

if hvd.local_rank() == 0:
    verbose = 2
else:
    verbose = 0

model.fit(dataset, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
          callbacks=callbacks, verbose=verbose)

if verbose > 0:
    epoch_times = time_history.epoch_times[1:]
    first_epoch = int(time_history.epoch_times[0])
    avg_epoch_time = sum(epoch_times)/len(epoch_times)
    avg_fps = round(hvd.size()*args.batch_size*steps_per_epoch/avg_epoch_time, 1)
    nv_stats_recorder = nv_stats.recorder
    gpu_data = nv_stats_recorder.get_data()
    device_data = gpu_data["device_data"]
    data_len = len(gpu_data["time_history"][first_epoch:])
    avg_sm = int(sum(gpu_data["sm_util_history"][first_epoch:])/data_len)
    avg_mem = int(sum(gpu_data["mem_util_history"][first_epoch:])/data_len)
    avg_pcie = int(sum(gpu_data["pcie_txrx"][first_epoch:])/data_len)
    pcie_gbps = round(sum(gpu_data["pcie_txrx"][first_epoch:])/data_len/100*device_data["max_pcie_bandwidth"]/1e6, 1)
    avg_pwr = int(sum(gpu_data["pwr_history"][first_epoch:])/data_len)
    pwr_watts = int(sum(gpu_data["pwr_history"][first_epoch:])/data_len/100*device_data["max_power"]/1e3)
    avg_temp = int(sum(gpu_data["temp_history"][first_epoch:])/data_len)
    max_vram = round(max(gpu_data["mem_occupy_history"])/100*device_data["total_vram"], 1)
    throttle = []
    for t in gpu_data["throttle_reasons"]:
        if t[1] > first_epoch:
            throttle.append(t[0])
    throttle = list(set(throttle))
    
    if RECORD_NVLINK:
        nvlink_stats_recorder = nvlink_stats.recorder
        skip_time = int(time_history.epoch_times[0]/nvlink_interval)
        nvlink_history = nvlink_stats_recorder.get_data()["nvlink_history"][skip_time:]
        print(nvlink_history)
        avg_nvlink_list = []
        for t in nvlink_history:
            avg_nvlink_list.append(
                sum([i for i in t.values()])/len(list(t.keys()))
            )
        avg_nvlink = round(sum(avg_nvlink_list)/len(avg_nvlink_list), 1)
    else:
        avg_nvlink = 0.0
    
    print("Results:")
    result_data = [
        "PASS", avg_fps, avg_sm, avg_mem, avg_pcie, pcie_gbps, avg_pwr, pwr_watts, avg_temp, max_vram, avg_nvlink, throttle
    ]
    results = ",".join([str(r) for r in result_data])
    print(results)
    