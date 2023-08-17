# computer-vision-store

## A. Installation

Please praprae this `anaconda`, `CUDA`, `CuDNN`, and `HuggingFace AccessToken` for practice.<br>
Follow this step!!

- [A.1. Install Anacdona for practice](./README.md#a1-install-anacdona-for-practice)
- [A.2. Check Graphic Card Name](./README.md#a2-check-graphic-card-name)
- [A.3. Install NVIDIA DRIVER](./README.md#a3-install-nvidia-driver)
- [A.4. Check Compatibility](./README.md#a4-check-compatibility)
- [A.5. Install Python@3.8.16 Depdencies](./README.md#a5-install-python3816-depdencies)
- [A.6. Some modules](./README.md#a6-some-modules)

### A.6. Some modules

### A.1. Install Anacdona for practice

When you're trying computer vision, you need to install some dependency, not supported windows.<br>
And then, you can use [anaconda](https://www.anaconda.com/).<br>
I used `conda 23.1.0`.

- Install environment

```cmd
conda env create -f requirements-conda.txt
```

- Export environment

```cmd
conda env export > requirements-conda.txt
```

### A.2. Check Graphic Card Name

1. Check [Windwos] - [Device Administrators] - [Display Adaptor]
2. Disable Basic GPU Instance supplied by CPU.
3. Notes `NVIDIA GeForce RTX 3080 Laptop GPU` to install NVIDIA Driver

<image style="width: 300px;" src="./images/docs/01_check-graphics-card-name.png"/>

### A.3. Install NVIDIA DRIVER

1. Install NVIDIDA DRIVER for `NVIDIA GeForce RTX 3080 Laptop GPU` or your instances.

- link :  https://www.nvidia.com/download/index.aspx

<image style="width: 600px;" src="./images/docs/02_install-nvidia-drvier.png"/>

### A.4. Check Compatibility

1. Check Graphics Card Compatibility among `NVIDIA GeForce RTX 3080 Laptop GPU` and **CUDA Computing Power** <br>
    -> `NVIDIA GeForce RTX 3080` supports [CUDA Computing Function 8.6](https://en.wikipedia.org/wiki/GeForce_30_series#cite_note-34)

2. Check Python 3.8.16 Compatibility among CUDA and CUDDN <br>
    -> [Deprecation of CUDA 11.6 and Python 3.7 Support](https://pytorch.org/blog/deprecation-cuda-python-support/)

| PyTorch Version   |	Python	|   Stable CUDA     |   Experimental CUDA |
| ----------------- | -------- | ------------ | ---------------------------- |
| 2.0	|   >=3.8, <=3.11	|   CUDA 11.7, CUDNN 8.5.0.96	|   CUDA 11.8, CUDNN 8.7.0.84 |
| 1.13	|   >=3.7, <=3.10	|   CUDA 11.6, CUDNN 8.3.2.44	|   CUDA 11.7, CUDNN 8.5.0.96 |
| 1.12	|   >=3.7, <=3.10	|   CUDA 11.3, CUDNN 8.3.2.44	|   CUDA 11.6, CUDNN 8.3.2.44 |

- [CUDA Toolkit 11.7 Downloads](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network)
- [Download cuDNN v8.5.0 (August 8th, 2022), for CUDA 11.x](https://developer.nvidia.com/rdp/cudnn-archive)

### A.5. Install Python@3.8.16 Depdencies

- Install environment

```cmd
pip install -r requirements-pip.txt
```

- Export environment

```cmd
pip freeze >> requirements-pip.txt
```

### A.6. Some modules

Some Module can't install without HuggingFace AccessToken.
Please signup and publish AccessToken to Downlaod these files....

- https://huggingface.co/pyannote/speaker-diarization
- https://huggingface.co/pyannote/segmentation

## B. Studies

- [OpenCV and Computer Vision](https://github.com/unchaptered/opencv-and-computer-vison)
- [OpenCV and Computer Vision Advanced](https://github.com/unchaptered/opencv-and-computer-vison-advanced)