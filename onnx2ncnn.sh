#!/bin/bash

../ncnn/build-host-gcc-linux/tools/onnx/onnx2ncnn checkpoints/mobilenet_v2-sim.onnx \
checkpoints/mobilenet_v2.param \
checkpoints/mobilenet_v2.bin 

../ncnn/build-host-gcc-linux/tools/ncnnoptimize \
checkpoints/mobilenet_v2.param \
checkpoints/mobilenet_v2.bin \
checkpoints/mobilenet_v2-opt.param \
checkpoints/mobilenet_v2-opt.bin \
65536