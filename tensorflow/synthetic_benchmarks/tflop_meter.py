import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_gpus", type=int, default=1,
                    help="Number of GPUs to use")
parser.add_argument("--stats", action="store_true", default=False,
                    help="Record stats using NVStatsRecorder")
parser.add_argument("--msize", default=5120, type=int,
                    help="Matrix size (M*M)")
parser.add_argument("--time", default=360, type=int,
                    help="Min time to continuously run the stress test")
args = parser.parse_args()

import os
import multiprocessing
n_cores = multiprocessing.cpu_count()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = str(n_cores)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import time
from tqdm import tqdm
import tensorflow as tf
tf.config.experimental.set_synchronous_execution(False)
tf.config.optimizer.set_jit(True)
gpu_list = tf.config.list_physical_devices("GPU")[:args.num_gpus]
print("GPUs:", gpu_list)
num_gpus = args.num_gpus
tf.config.threading.set_inter_op_parallelism_threads(n_cores)
    
if args.stats:
    from nvstatsrecorder.recorders import NVStatsRecorder

@tf.function(experimental_compile=True)
def do_op(a, b):
    return tf.linalg.matmul(a, b)

def benchmark_matmul(M, gpus=1, dtype=tf.float32, iterations=100):
    # generate data and warm-up iteration
    slots = []
    for i in range(num_gpus):
        with tf.device("/GPU:"+str(i)):
            A = tf.random.normal([M, M], mean=0, stddev=1, dtype=dtype)
            B = tf.random.normal([M, M], mean=0, stddev=1, dtype=dtype)
            C = do_op(A, B)
            slots.append((A, B))
    for i in range(num_gpus):
        with tf.device("/GPU:"+str(i)):
            C = do_op(slots[i][0], slots[i][1])
    C.numpy()
    # measure overhead
    st = time.time()
    for _ in range(1):
        for i in range(num_gpus):
            with tf.device("/GPU:"+str(i)):
                C = do_op(slots[i][0], slots[i][1])
    C.numpy()
    et = time.time()
    overhead = et - st
    # run benchmark
    st = time.time()
    for _ in range(iterations+1):
        for i in range(num_gpus):
            with tf.device("/GPU:"+str(i)):
                C = do_op(slots[i][0], slots[i][1])
    C.numpy()
    et = time.time()
    duration = (et-st) - overhead
    return gpus*iterations/duration

if args.stats:
    try:
        SUDO_PASSWORD = os.environ["SUDO_PASSWORD"]
        nv_stats_recorder = NVStatsRecorder(gpu_index=0, tensor_util=True, sudo_password=SUDO_PASSWORD)
    except:
        nv_stats_recorder = NVStatsRecorder(gpu_index=0)
    nv_stats_recorder.start(interval=3)
    
fp16_tflops, fp32_tflops, fp64_tflops = [], [], []

M = args.msize
iterations = 100

print("\nStarting burn...\n")

burn_start = time.time()

while time.time()-burn_start < args.time:
    print("FP64", M, end=" : ")
    ret = benchmark_matmul(M, gpus=num_gpus, dtype=tf.float64, iterations=iterations)
    tflops = ret * 2 * M**3 / 1e12
    fp64_tflops.append(tflops)
    print(tflops)
    print("FP32", M, end=" : ")
    ret = benchmark_matmul(M, gpus=num_gpus, dtype=tf.float32, iterations=iterations)
    tflops = ret * 2 * M**3 / 1e12
    fp32_tflops.append(tflops)
    print(tflops)
    print("FP16", M, end=" : ")
    ret = benchmark_matmul(M, gpus=num_gpus, dtype=tf.float16, iterations=iterations)
    tflops = ret * 2 * M**3 / 1e12
    fp16_tflops.append(tflops)
    print(tflops)
    done = int((time.time()-burn_start)/args.time*100)
    print("Done:", done, "%")
    
burn_end = time.time()
    
print("\nFinished in", int(burn_end-burn_start), "seconds\n")

num_gpus = str(num_gpus)
    
if args.stats:
    nv_stats_recorder.stop()
    nv_stats_recorder.plot_gpu_util(smooth=3, outpath="burn_"+num_gpus+"_gpu_util.jpg")
    nv_stats_recorder.plot_gpu_temp(smooth=3, outpath="burn_"+num_gpus+"_gpu_temp.jpg")
    
title = "Max TFLOPS achieved (" + num_gpus + " GPUs)"
print("")
print(title)
print("="*len(title))
print("* FP64:", int(max(fp64_tflops)), "TFLOPS")
print("* FP32:", int(max(fp32_tflops)), "TFLOPS")
print("* FP16:", int(max(fp16_tflops)), "TFLOPS")
print("")
