# PyTorch Benchmarks

## Small vGPU Benchmark

This is a test designed to stress of a vGPU instance with the CPU/GPU equivalent to 1/64 DGX-1V (1.2 thread, 1/8 V100 with 4GB HBM2). The test is expected to take about 6GB RAM and 3GB VRAM.

```
python3 vgpu_test/cifar10.py
```

Result (on Titan V)

```
Finished training!
==================
Average images/sec: 3158
Model saved to ./cifar_net.pth
End-to-end time: 161
```
