# RecON

This repository is the official implementation for "[RecON: Online Learning for Sensorless Freehand 3D Ultrasound Reconstruction](https://doi.org/10.1016/j.media.2023.102810)".

## Environment
- PyTorch with GPU
- OpenCV-Python build from CUDA
- Run `pip install -r requirements.txt`

## Training
- Backbone
    ```shell
    python3 -m main -m online_bk -d Spine -r hp_bk -g0
    ```
- Discriminator
    ```shell
    python3 -m main -m online_d -d Spine -r hp_d -g0
    ```

## Online Learning
```shell
python3 -m main -m online_fm -d Spine -r hp_fm -g0 -t0
```

## Demo
An interactive demo is available in [here](http://apps.myluo.cn/RecON).