
# SpanMarkerNER on Triton-TensorRT

Quick guide to run SpanMarkerNER on the TensorRT backend of the Triton Inference Server!

## Hardware

Instance `g5.48xlarge` with AL2023 AMI and NVIDIA drivers installed as follows https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html#nvidia-GRID-driver leading to
```
[ec2-user@ip-172-31-33-234 ~]$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
```

## Prerequisites

Run the Convert-SpanMarker-to-ONNX notebook, to obtain `model.onnx`

Create a model repository like the following
```
triton_model_repository
`-- spanmarker
    |-- 1
    |   |-- model.onnx
    `-- config.pbtxt
```
where `config.pbtxt` looks like the following
```
name: "spanmarker"
backend: "tensorrt"
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ 1, 256 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_BOOL
    dims: [ 1, 256, 256 ]
  },
  {
    name: "position_ids"
    data_type: TYPE_INT64
    dims: [ 1, 256 ]
  },
  {
    name: "num_marker_pairs.1"
    data_type: TYPE_INT64
    dims: [ 1 ]
  },
  {
    name: "num_words.1"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }
]
output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, 64, 5 ]
  },
  {
    name: "num_marker_pairs"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "num_words"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]
```
Ensure dimensions are aligned with the ONNX obtained in the step before.

## Setup

### Install docker 

Source: https://docs.docker.com/engine/install/rhel/


### Enable GPU for docker

Source: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Then
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Pull docker container

Source: https://github.com/aws/deep-learning-containers/blob/master/available_images.md#nvidia-triton-inference-containers-sm-support-only

For instance
```
export ACCOUNT_ID="763104351884"
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tritonserver:25.04-py3
```

### Run docker container

Source: https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_1-model_deployment#setting-up-the-model-repository

For instance
```
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/models/optim-span-marker-xlm-roberta-base-conll2003/onnx/triton_model_repository:/models 763104351884.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tritonserver:25.04-py3
```

### Run triton inference server
Compile the model to tensorrt, ensure dimensions are aligned with the config above!
```
/usr/src/tensorrt/bin/trtexec \
  --onnx=/models/spanmarker/1/model.onnx \
  --saveEngine=/models/spanmarker/1/model.plan \
  --minShapes=input_ids:1x256,attention_mask:1x256x256,position_ids:1x256,num_marker_pairs.1:1,num_words.1:1 \
  --optShapes=input_ids:1x256,attention_mask:1x256x256,position_ids:1x256,num_marker_pairs.1:1,num_words.1:1 \
  --maxShapes=input_ids:1x256,attention_mask:1x256x256,position_ids:1x256,num_marker_pairs.1:1,num_words.1:1
```
Start the server!
```
tritonserver --model-repository=/models
```

## Performance analyzer
```
docker pull nvcr.io/nvidia/tritonserver:24.12-py3-sdk
```

and then
```
docker run -ti --rm --gpus=all --network=host -v $PWD:/mnt --name triton-client nvcr.io/nvidia/tritonserver:24.12-py3-sdk
```
```
perf_analyzer -m spanmarker --request-rate-range 10:50:10
```
At the end, you should see something like
```
Inferences/Second vs. Client Average Batch Latency
Request Rate: 10, throughput: 9.99449 infer/sec, latency 3704 usec
Request Rate: 20, throughput: 19.9762 infer/sec, latency 3688 usec
Request Rate: 30, throughput: 29.9609 infer/sec, latency 3683 usec
Request Rate: 40, throughput: 39.9427 infer/sec, latency 3647 usec
Request Rate: 50, throughput: 49.9861 infer/sec, latency 3674 usec
```
~3.6ms of latency, for a model with 278,055,941 parameters, no quantization and poorly optimized sequence length, not bad!