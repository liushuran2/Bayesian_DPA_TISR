# Bayesian_DPA_TISR:Time-lapse Image Super-resolution Neural Network with Reliable Confidence Evaluation for Optical Microscopy

Code, representive pretrain model, test data of Bayes_DPA_TISR
<div align="center">

✨ [**Method**](#-Method-overview) **|** 🚩 [**Paper**](#-Paper) **|** 🔧 [**Install**](#Install)  **|** 🎨 [**Dataset**](#-Dataset)  **|** 💻 [**Training**](#-Training) **|** 🏰 [**Model Zoo**](#-Model-Zoo)  **|** ⚡ [**Inference**](#-Inference) **|** &#x1F308; [**Results**](#-Results)

</div>

## ✨ Method overview

<p align="center">
<img src="assert\fig1.jpg" width='600'>
</p>
We first build a large-scale, high-quality dataset for the time-lapse image super-resolution (TISR) task, and conducted a comprehensive evaluation on two essential components, i.e., propagation and alignment mechanisms, of TISR methods. Second, we devised the deformable phase-space alignment (DPA) based TISR neural network (DPA-TISR), which adaptively enhances the cross-frame alignment in the phase domain and outperforms existing state-of-the-art TISR models. 
<p align="center">
<img src="assert\fig2.jpg" width='600'>
</p>
Third, we combined the Bayesian training scheme with DPA-TISR, dubbed Bayesian DPA-TISR, and designed an expected calibration error (ECE) minimization framework to obtain a well-calibrated confidence map along with each output SR image, which reliably implicates potential inference errors. We demonstrate that the Bayesian DPA-TISR achieves resolution enhancement by more than 2-fold compared to diffraction limits with high fidelity and temporal consistency, enabling confidence-quantifiable TISR in long-term live-cell SR imaging for various bioprocesses.

## 🚩 Paper
This repository is for Bayesian DPA-TISR introduced in the following paper:

[Chang Qiao, Shuran Liu, Yuwang Wang, et al. "Time-lapse Image Super-resolution Neural Network with Reliable Confidence Evaluation for Optical Microscopy." ***bioRxiv 2024.05.04.592503*** (2024)](https://doi.org/10.1101/2024.05.04.592503) 

## 🔧 Install
### Our environment
  - Ubuntu 20.04.5
  - CUDA 11.3.1
  - Python 3.8.13
  - Pytorch 1.12.1
  - NVIDIA GPU (GeForce RTX 3090) 

### Quick installation
If you have already set up a PyTorch environment, no matter using conda or pip, you can start from **step 3**.

**Step 0.**
Download and install Anaconda from [official website](https://www.anaconda.com/download/).

**Step 1.**
Clone this repository using the following command.
   ```bash
    git clone https://github.com/liushuran2/Bayesian_DPA_TISR.git
  ```

**Step 2.**
Create a virtual environment and install PyTorch. **If your CUDA maximum version is < 11.3**, please select the correct Pytorch version that matches your CUDA version from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/). The time for installing is usually less than 10 minutes.

 ```bash
    $ conda create -n DPATISR python=3.8
    $ conda activate DPATISR
    $ pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
**Step 3.**
To install other dependencies used in our code, simply run the following command.
 ```bash
    $ pip install -r requirements.txt
```

### Version of CUDA

When installing PyTorch in **Step 2**, you need to specify the version of CUDA.
If you are not clear on which to choose, follow our recommendations:

1. For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, **CUDA 11 is a must**.
2. For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please also make sure the GPU driver satisfies the minimum version requirements.
See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

### Install without Conda

Though we highly recommend using conda to create environments and install PyTorch, it is viable to install PyTorch only with pip, for example, with the following command,

```shell
pip3 install torch torchvision
```

However, an `--extra-index-url` or `-f` option is usually required to specify the CPU / CUDA version.
See [PyTorch website](https://pytorch.org/get-started/locally/) for more details.
## 🏰 Model Zoo
| Scale factor    | Cell Structure  |Download                |
| ---------------|:--------- | :------------------------------------------- |
| 2  | Microtubules  |  [Zenodo repository](https://doi.org/10.5281/zenodo.12593629)                                              |
| 3 | F-actin     |    [Zenodo repository](https://doi.org/10.5281/zenodo.12593629)  
| 2  | Mitochondria    |    [Zenodo repository](https://doi.org/10.5281/zenodo.12593629)  

Download the pre-trained models and place them into `./checkpt`.


## ⚡ Inference

### 1. Prepare models

Before inference, you should have trained your own model or downloaded our pre-trained model. 

### 2. Edit confiuration
Before inference with pre-trained model, please carefully edit the [config.yaml](https://github.com/liushuran2/Bayesian_DPA_TISR/blob/main/config.yaml) file. 
* Change the *inference_checkpt* line to the actual path of pre-trained model. 
* Change *test_dataset_path* line to the actual path of test data(h5 file, detailed in [**Dataset**](#-Dataset),eg `./dataset/F-actin.h5`).
* Other key confiurations are *bayesian* and *factor* etc. Please keep them the same as in training or pretraining.

### 3. Test models
```python
python test.py --config config.yaml
```

* The "config" sets the training configuration file path.
* The super-resolution and confidence results will be saved in `./results`.

## 💻 Training 

### 1. Prepare the data 
You can use your own data or download BioTISR below(detailed in [**Dataset**](#-dataset)). 

### 2. Edit confiuration
Before training, please carefully edit the [config.yaml](https://github.com/liushuran2/Bayesian_DPA_TISR/blob/main/config.yaml) file. Some **must-change** parameters are as follows:
* *train_dataset_path* and *valid_dataset_path* (the path of your training data and validation data(*.h5 file, detailed in [**Dataset**](#-Dataset))
* *checkpoint_folder* and *checkpoint_name* (the location that checkpoint will be saved in)
* *tensorboard_folder* (the location that tensorboard log file will be saved in for visualization)

Other key parameter are presented in [config.yaml](https://github.com/liushuran2/Bayesian_DPA_TISR/blob/main/config.yaml) file with , do not change if you don't know what it is.

### 3. Start training
Simply run:
```python
python train.py --config config.yaml
```

The visualization of training procedure is also provided. Run the following command:
```python
tensorboard --logdir=tensorboard --port=6006 --host='localhost'
```

### 4. Confidence correction
Aiming at overcoming the commonly overconfidence problem, user can adopt the algorithm mentioned in our paper to minimize Expected calibration error (ECE) by running:
```python
python finetune.py --config config.yaml
```

## 🎨 Dataset
We acquired an extensive TISR dataset (BioTISR), of five different biological structures: clathrin-coated pits (CCPs), lysosomes (Lyso), outer mitochondrial membranes (Mito), microtubules (MTs), and F-actin filaments.

BioTISR is now freely available, aiming to provide a high-quality dataset for the community of time-lapse bio-image super-resolution algorithm and advanced SIM reconstruction algorithm developers.

Scripts for reading **MRC** file are provided with the dataset. 

In this repository, you can find a script named [read_mrc.py](https://github.com/liushuran2/Bayesian_DPA_TISR/blob/main/prepare_data.py) which organizes seperate images into an [hdf5](https://www.hdfgroup.org/solutions/hdf5/) file used in the training and testing for faster loading speed. After changing the loading and saving path, simply run:
```python
python read_mrc.py
```

## &#x1F308; Results

 ### 1. Comparison with other SOTA model.

<p align="center">
<img src="assert\fig3.jpg" width='600'>
</p>

### 2. Confidence calculation and correction for DPA-TISR.

<p align="center">
<img src="assert\fig4.jpg" width='600'>
</p>

### 3. Long-term SR live imaging with reliable confidence evaluation via Bayesian DPA-TISR.

<p align="center">
<img src="assert\fig5.jpg" width='600'>
</p>
