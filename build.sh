#! /bin/bash

apt-get install -y nvidia-docker2

docker pull pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

nvidia-docker run -it -d --name="onnx2pytorch" -v  ./:/workspace pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

docker exec -it onnx2pytorch /bin/bash