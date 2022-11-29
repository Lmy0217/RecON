# RecON: Online Learning for Sensorless Freehand 3D Ultrasound Reconstruction

This repository contains the code for the paper "RecON: Online Learning for Sensorless Freehand 3D Ultrasound Reconstruction".

## Environment
* PyTorch with GPU
* OpenCV-Python build from CUDA
* Run ` pip inatall -r requirements.txt`

## Pre-train models
Our pre-trained models are available in [release](https://github.com/Lmy0217/RecON/releases).

## Testing
```shell
python -m main -m online_fm -d Spine -r hp_fm -g0 -t0
```

## Demo
An interactive demo is available in [here](http://apps.myluo.cn/RecON).