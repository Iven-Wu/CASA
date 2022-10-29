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
We provide instructions for shape optimization on synthetic data.

## Preparing Dataset

We use the Planetzoo dataset. 

Download the [Planetzoo Dataset]()

## Retrieval
We provide encoded features in `./dataset/embeddings`

If you only want to try retrieval part, use the code below.

```
python clip_retrieve.py 
```




## Optimization

Optimize the shape, skeleton, skinning weight parameters from observations.

```
python main.py -c config/synthetic.yaml
```



<!-- <details><summary>Real World Data</summary>
Similarly, run the following steps to reconstruct pika

```
python main.py -c config/real_world.yaml
```

</details> -->


<!-- ## Acknowledgments
Our code is built on [LASR](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020). Thank the authors for sharing their codes! -->


<!-- Mesh download [simplified meshes](https://drive.google.com/drive/folders/1g8RMN_MLN2ZOlbiy8j6pPk3GUZYI8DLG?usp=sharing) -->
