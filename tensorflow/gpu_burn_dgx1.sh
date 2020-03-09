#!/usr/bin/env bash

echo "Start script"

echo "Run 8 GPU, with stats"
python3 gpu_burn.py --num_gpus 8 --stats

sleep 1

echo "Run 8 GPU, without stats"
python3 gpu_burn.py --num_gpus 8

sleep 1

echo "Run 1 GPU, with stats"
CUDA_VISIBLE_DEVICES=0 python3 gpu_burn.py --num_gpus 1 --stats

sleep 1

echo "Run 1 GPU, without stats"
CUDA_VISIBLE_DEVICES=0 python3 gpu_burn.py --num_gpus 1

echo "All done!"
