Embedded AI Torch to TFLite on ESP32
===

We provide an example to train a Conv model with Pytorch 
and convert to Tensorflow Lite (Micro) and then deploy it
on ESP32-S3-DevKitC-1-N8R8.

## Qucik start
You can play the notebook 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/nanguoyu/Embedded-AI-Torch-to-TFLite-on-ESP32/)
in Google Colab.

## How does it work?

There is a gap between deep models and resource-limited devices.
From Kb-level microcontroller to powerful Edge devices 
with hardware accelerator, the on-board resource varies.
It requires model compression and lightweight model 
designs.

We take the deployment of a simple CNN model on ESP32
as an example. We implement a tiny CNN model with Pytorch
and train it on MNIST for epochs, and then convert it to 
ONNX and then take `ONNX-Tensorflow` to perform conversion 
and full-integer quantization. The well-quantized model 
is exported to a C array containing in a `.CC` model file.

To deploy TFlite model to ESP32, we implement an inference 
program for ESP32-S3 and flash it into board with 
`ESP-IDF`. 

(We will release this part soon)

