# Deployment Scripts

## Inference with Python TensorRT

Export ONNX model
```bash
python deployment_scripts/export_onnx.py
```
Build TensorRT engine
```bash
bash deployment_scripts/build_engine.sh
```
Inference with TensorRT
```bash
python deployment_scripts/gr00t_inference.py --inference_mode=tensorrt
```

---

## Jetson Deployment

### Prerequisites

- AGX Orin installed with Jetpack 6.2

### 1. Installation Guide

Clone the repo:

```sh
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

### Install Isaac-GR00T directly

Run below setup script to install the dependencies:

```sh
bash deployment_scripts/setup_env.sh
```

### Deploy Isaac-GR00T with Container

#### Build Container

To build a container for Isaac-GR00T:

```sh
docker build -t isaac-gr00t-n1.5:l4t-jp6.2 -f orin.Dockerfile .
```

#### Run Container

To run the container:
```sh
docker run -it --rm --network=host --privileged --runtime=nvidia \
    -v /media/binliu/mxgu/Isaac-GR00T:/mnt/Isaac-GR00T \
    -v /media/binliu/BLSSD/trt/gr00t_engine:/mnt/Isaac-GR00T/gr00t_engine \
    -v /media/binliu/BLSSD/trt/gr00t_onnx:/mnt/Isaac-GR00T/gr00t_onnx \
    -v /media/binliu/mxgu/lerobot:/mnt/lerobot \
    -v /dev:/dev \
    --workdir /mnt/Isaac-GR00T \
    -v /media/binliu/BLSSD/checkpoints/:/mnt/Isaac-GR00T/checkpoints \
    -v /media/binliu/mxgu/so101_follower_arm.json:/root/.cache/huggingface/lerobot/calibration/robots/so101_follower/so101_follower_arm.json \
    isaac-gr00t-n1.5:l4t-jp6.2  /bin/bash
```

```sh
# lerobot
pip install -e ".[feetech]"
```

```sh
# gr00t
pip install -e .[orin]
```

```sh
export PYTHONPATH=/mnt/Isaac-GR00T:$PYTHONPATH
```

```sh
python scripts/inference_service_trt.py --server \
    --embodiment-tag new_embodiment \
    --data-config so100_dualcam \
    --denoising-steps 4 \
    --model_path /mnt/Isaac-GR00T/checkpoints/checkpoint-30000/

# inference request, scissor
python getting_started/examples/eval_lerobot.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=so101_follower_arm \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, room: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --policy_host=127.0.0.1 \
    --lang_instruction="Grip the scissors and put it into the tray"
```

### 2. Inference

* The GR00T N1.5 model is hosted on [Huggingface](https://huggingface.co/nvidia/GR00T-N1.5-3B)
* Example cross embodiment dataset is available at [demo_data/robot_sim.PickNPlace](./demo_data/robot_sim.PickNPlace)
* This project supports to run the inference with PyTorch or Python TensorRT as instructions below

### 2.1 Inference with PyTorch

```bash
python deployment_scripts/gr00t_inference.py --inference_mode=pytorch
```

### 2.2 Inference with Python TensorRT

Export ONNX model
```bash
python deployment_scripts/export_onnx.py
```
Build TensorRT engine
```bash
bash deployment_scripts/build_engine.sh
```
Inference with TensorRT
```bash
python deployment_scripts/gr00t_inference.py --inference_mode=tensorrt
```

## 3. Performance
### 3.1 Pipline Performance
Here's comparison of E2E performance between PyTorch and TensorRT on Orin:

<div align="center">
<img src="../media/orin-perf.png" width="800" alt="orin-perf">
</div>

### 3.2 Models Performance
Model latency measured by `trtexec` with batch_size=1.     
| Model Name                                     |Orin benchmark perf (ms)  |Precision|
|:----------------------------------------------:|:------------------------:|:-------:|
| Action_Head - process_backbone_output          | 5.17                     |FP16     |
| Action_Head - state_encoder                    | 0.05                     |FP16     |
| Action_Head - action_encoder                   | 0.20                     |FP16     |
| Action_Head - DiT                              | 7.77                     |FP16     |
| Action_Head - action_decoder                   | 0.04                     |FP16     |
| VLM - ViT                                      |11.96                     |FP16     |
| VLM - LLM                                      |17.25                     |FP16     |  
      
**Note**：The module latency (e.g., DiT Block) in pipeline is slighly longer than the modoel latency in benchmark table above because the module (e.g., Action_Head - DiT) latency not only includes the model latency in table above but also accounts for the overhead of data transfer from PyTorch to TRT and returning from TRT to to PyTorch.
