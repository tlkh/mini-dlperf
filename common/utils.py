import os
import time
import tensorflow as tf

        
def run_command(command):
    stream = os.popen(command)
    raw_output = stream.read().split("\n")
    output = []
    for o in raw_output:
        if "<stdout>:" in o:
            o = o.split("<stdout>:")[-1]
        if len(o) > 0:
            output.append(o)
    return output


def get_gpu_info():
    command = """python3 -c "import tensorflow as tf; tf.config.list_physical_devices('GPU')" > /tmp/gpu_info.txt 2>&1"""
    run_command(command)
    lines_raw = [line.rstrip('\n') for line in open("/tmp/gpu_info.txt")]
    gpus = {}
    for i, l in enumerate(lines_raw):
        if "pciBusID: " in l:
            key = l.split(" computeCapability: ")[0]
            key = key.replace("pciBusID: ", "").replace(" name: ", ",")
            compute_capability = "computeCapability: " + l.split(" computeCapability: ")[1]
            gpu_spec = compute_capability + " " + lines_raw[i+1]
            gpus[key] = gpu_spec
    return gpus

        