# CASA
This repository contains the official implementation for VideoINR introduced in the following paper:

[**CASA: Category-agnostic Skeletal Animal
Reconstruction**](https://Iven-Wu.github.io/CASA)
<br>
[Yuefan Wu*](http://ivenwu.com/), [Zeyuan Chen*](https://zeyuan-chen.com/), [Shaowei Liu](https://stevenlsw.github.io/), [Zhongzheng Ren](https://jason718.github.io/),  [Shenlong Wang](http://shenlong.web.illinois.edu/)
<br>
NeurIPS 2022

You can find more visual results and a brief introduction to CASA at our [project page](https://Iven-Wu.github.io/CASA).


## Environmental Setup

The code is tested in:
- Python 3.8
- Pytorch 1.11.0
- torchvision 0.12.0
- Cuda 11.3
- [Lietorch](https://github.com/princeton-vl/lietorch)
- [Chamfer3D](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)
- [SoftRas](https://github.com/ShichenLiu/SoftRas). Following implementation in [LASR](https://github.com/google/lasr)

If you are using Anaconda, the following command can be used to build the environment:


```
conda env create -f casa.yml
conda activate casa

# install softras
# to compile for different GPU arch, see https://discuss.pytorch.org/t/compiling-pytorch-on-devices-with-different-cuda-capability/106409

pip install -e softras
```

## Overview
We provide instructions for shape optimization on two types of data,
- Synthetic: Video frames from Planetzoo Dataset.
- Real World: In the wild video frames.

We recomend first trying sythetic first.

## Preparing Dataset

We use the [Planetzoo dataset](https://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/). 

Download the [Planetzoo Dataset]()

## Retrieval
We provide encoded features in `./dataset/embeddings`
<details><summary>Synthetic Data</summary>

```
python clip_retrieve_cus.py 
```
</details>

<details><summary>Real World Data</summary>

You will need to install and clone [detectron2](https://github.com/facebookresearch/detectron2) to obtain instance segmentations.

```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.11/index.html
git clone https://github.com/facebookresearch/detectron2
```

```
python mask.py
python clip_retrieve_real.py 
```
</details>

## Optimization

<details><summary>Synthetic Data</summary>
Next, we want to optimize the shape, skeleton, skinning weight parameters from observations.

```
python optimize_final.py
```

</details>


<details><summary>Real World Data</summary>
Similarly, run the following steps to reconstruct pika

```
python optimize_final.py
```

</details>


<!-- ## Acknowledgments
Our code is built on [LASR](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020). Thank the authors for sharing their codes! -->


<!-- Mesh download [simplified meshes](https://drive.google.com/drive/folders/1g8RMN_MLN2ZOlbiy8j6pPk3GUZYI8DLG?usp=sharing) -->
