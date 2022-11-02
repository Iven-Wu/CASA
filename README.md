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


If you are using Anaconda, the following command can be used to build the environment:


```
conda env create -f casa.yml
conda activate casa

# install lietorch
git clone --recursive https://github.com/princeton-vl/lietorch.git
cd lietorch
python setup.py install
cd -

# install clip
pip install git+https://github.com/openai/CLIP.git

# install softras
# to compile for different GPU arch, see https://discuss.pytorch.org/t/compiling-pytorch-on-devices-with-different-cuda-capability/106409

pip install -e softras
```

## Overview
We provide instructions for shape optimization on synthetic data.

## Preparing Dataset

We use the Planetzoo dataset. 

Download the [Planetzoo Dataset]()

The dataset should have a structure as follows:
```
<data_source_name>/
    <animal_name>/

        ├── frame_000001.obj ### ground truth mesh
        ├── frame_000002.obj
        ├── ...
        ├── skeleton ### skeleton
        ├── weight   ### skinning weight
        ├── info    
            ├── 0001.npz ### mask,flow,camera,etc..
            ├── 0002.npz 
            ├── ...
            ├── 0001.png ### renderer images
            ├── 0002.png
            ├── ...

```
## Retrieval
We provide encoded features in `./dataset/embeddings`

If you only want to try retrieval part, use the code below.

```
python clip_retrieve.py 
```




## Optimization

Optimize the shape, skeleton, skinning weight parameters.

```
python main.py -c config/synthetic.yaml
```


## Acknowledgement
The code is built based on [LASR](https://github.com/google/lasr). Thank the authors for sharing their codes!

External repos:
- [Lietorch](https://github.com/princeton-vl/lietorch)
- [Chamfer3D](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)
- [CLIP](https://github.com/openai/CLIP)
- [SoftRas](https://github.com/ShichenLiu/SoftRas). Following implementation in [LASR](https://github.com/google/lasr)

## Citation

To cite our paper,
```
@article{wu2022casa,
    title={CASA: Category-agnostic Skeletal Animal Reconstruction},
    author={Yuefan Wu*, Zeyuan Chen*, Shaowei Liu, Zhongzheng Ren, Shenlong Wang},
    journal={NeurIPS},
    year={2022}
    }
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
