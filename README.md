# FaceVerse V4

## FaceVerse: a Fine-grained and Detail-controllable 3D Face Morphable Model from a Hybrid Dataset

This project is associated with the work of [Lizhen Wang](https://lizhenwangt.github.io/), Zhiyuan Chen, Tao Yu, Chenguang Ma, Liang Li, and [Yebin Liu](http://www.liuyebin.com/) presented at CVPR 2022. It is a collaboration between Tsinghua University and Ant Group.

- **[Dataset]**: [Link to Dataset](https://github.com/LizhenWangT/FaceVerse-Dataset)
- **[Project Page]**: [Project Page Link](http://www.liuyebin.com/faceverse/faceverse.html)
- **[Arxiv]**: [Arxiv Paper Link](https://arxiv.org/abs/2203.14057)

### Abstract
>We present FaceVerse, a fine-grained 3D Neural Face Model, which is built from hybrid East Asian face datasets containing 60K fused RGB-D images and 2K high-fidelity 3D head scan models. A novel coarse-to-fine structure is proposed to take better advantage of our hybrid dataset. In the coarse module, we generate a base parametric model from large-scale RGB-D images, which is able to predict accurate rough 3D face models in different genders, ages, etc. Then in the fine module, a conditional StyleGAN architecture trained with high-fidelity scan models is introduced to enrich elaborate facial geometric and texture details. Note that different from previous methods, our base and detailed modules are both changeable, which enables an innovative application of adjusting both the basic attributes and the facial details of 3D face models. Furthermore, we propose a single-image fitting framework based on differentiable rendering. Rich experiments show that our method outperforms the state-of-the-art methods.

## FaceVerse model and pre-trained checkpoints

This Git offers a faster and easy-to-use 3DMM tracking pipeline with **FaceVerse V4**, which is a full head model that includes separate eyeballs, teeth, and tongue.

<img src="example/input/exmaple.gif" width="480">

**Fig.1** FaceVerse version 4.

This work is based on the faceverse model and was trained on [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch) to obtain a ResNet50 that can directly predict parameters from images. The training process involved using the faceverse rendering dataset and the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset. The 621-Dimensional ResNet50 Output:

```
## Shape (156)
These define the core facial geometry, dictating the face's outline, nose shape, cheek curvature, and other feature details.

## Expression (177)
Encode facial expressions. The last 6 for tongue (untrained), with the other 171 capturing smiles, frowns, etc.

## Texture (251)
Specify face color and texture, including skin tone and markings.

## Gamma Param for Lighting (27)
Utilize SH for facial lighting, controlling light intensity, direction, and color to create realistic shadows and highlights.

## Rotation (3)
For x, y, z axes, orienting the face in 3D space, determining pitch, yaw, and roll.

## Translation (3)
2 for x, y translation and 1 for z (also handling scale in perspective projection).

## Eye Ball Rotations (4)
x, y for left and x, y for right eyes, defining eye orientations.
```

## Performance

The entire computation process involves using mediapipe to obtain a relatively stable face bounding box. Then, the ResNet50 is used to directly predict 621-dimensional parameters. Some smoothing operations are also applied. Finally, Sim3DR from [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) is utilized for rapid CPU rendering to obtain the final result.

The overall time consumption mainly includes the time for reading data, mediapipe processing, ResNet50 running, and data storage. In the webcam mode with a 480p input on an NVIDIA RTX 3060 laptop GPU, a frame rate of over 25fps has been achieved in practical tests.

## Requirements for v4

- Python 3.9
- PyTorch 2.5.1
- torchvision 0.20.1
- mediapipe 0.10.20
- Pillow 10.2.0
- opencv-python 4.10.0.84
- cython 3.0.11

## Usage

1. You need to download the following three files and place them in the `data/` folder:

- **FaceVerse version 4 model**: [[faceverse_v4_2.npy]](https://1drv.ms/u/c/b8eab7b1820a6fa4/EWJOsgGxPMZDkl8xJ_QZB30BpcjNoMVGK9mnUPq5n9-lyw?e=4GvEs9)
- **FaceVerse version 4 network**: [[faceverse_resnet50.pth]](https://1drv.ms/u/c/b8eab7b1820a6fa4/EcgUCYq20NhIqx-pGjnzDxkBfh9kMTtEn4G5UcOaHvwW4Q?e=eQnX64)
- **mediapipe face_landmarker.task**: [[face_landmarker.task]](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)


2. The next step is to set up the environment and compile the necessary components.

```
# install required libs
pip install -r requirements.txt

# compile the Sim3DR CPU renderer, easy for g++/gcc
# Windows need to install MSVC 14.0 and mv the rc.exe & rcdll.dll into the tool path
# or you can use the default Sim3DR_Cython.cp39-win_amd64.pyd compiled on Win11 py39
cd Sim3DR
python setup.py build_ext --inplace
cd..
```

3. Finally, you can start using the FaceVerse V4 for different input types.

```
# mp4 file
python run.py --input example/input/test.mp4 --output example/output --save_results True --smooth True
# image file
python run.py --input example/input/test.jpg --output example/output --save_results True --smooth False
# image folder
python run.py --input example/input/imagefolder --output example/output --save_results True --smooth False
# online face tracking with the webcamera
python run.py --input webcam --save_results False --smooth True
```

## Citation
If you use this dataset for your research, please consider citing:

```
@InProceedings{wang2022faceverse,
title={FaceVerse: a Fine-grained and Detail-controllable 3D Face Morphable Model from a Hybrid Dataset},
author={Wang, Lizhen and Chen, Zhiyua and Yu, Tao and Ma, Chenguang and Li, Liang and Liu, Yebin},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR2022)},
month={June},
year={2022},
}
```

## Contact
- Lizhen Wang [(wanglz@126.com)](wanglz@126.com)

## Acknowledgement & License
The code is partially borrowed from [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch) (networks), [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) (Sim3DR). We are also grateful to the volunteers who participated in data collection. Our License can be found in [LICENSE](./LICENSE).