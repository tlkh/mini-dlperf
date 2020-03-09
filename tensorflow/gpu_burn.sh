#!/usr/bin/env bash

echo "Start script"

echo "Run 4 GPU, with stats"
python3 gpu_burn.py --num_gpus 4 --stats

sleep 1

echo "Run 4 GPU, without stats"
python3 gpu_burn.py --num_gpus 4

sleep 1

echo "Run 1 GPU, with stats"
CUDA_VISIBLE_DEVICES=3 python3 gpu_burn.py --num_gpus 1 --stats

sleep 1

echo "Run 1 GPU, without stats"
CUDA_VISIBLE_DEVICES=3 python3 gpu_burn.py --num_gpus 1

echo "All done!"
