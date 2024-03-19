## FFNet: Simple and Efficient Architectures for Semantic Segmentation
FFNets are families of Simple and Efficient Architectures, which we demonstrate the effectiveness of for the task of Semantic Image Segmentation.
This repository provides the model definitions and pre-trained weights for the models introduced in the paper [Simple and Efficient Architectures for Semantic Segmentation](https://arxiv.org/abs/2206.08236), published at the Efficient Deep Learning for Computer Vision Workshop at CVPR 2022.


FFNet stands for "Fuss-Free Networks", and utilize a simple ResNet-like backbone, and a tiny convolution-only head to produce multi-scale features that are useful for various tasks. 
![FFNet 4-stage](figures/ffnet_architecture.png?raw=true "Architecture of FFNets, with a 4-stage ResNet like backbone comprised of basic-blocks, and a tiny convolution-only head.")

Our key takeaway is that when comparing with various architectures and approaches with a tad more `fuss', ResNet based approaches were being put at a massive disadvantage due to the severly limited receptive fields of ResNet-50/101, owing to the use of bottleneck blocks. While this problem has been acknowledged in prior work, attempts to remedy it have typically involved the use of dilated convolutions to increase the receptive field. Dilated convolutions tend to be slow on current hardware. We show that using deep ResNets with basic-blocks as the backbone/encoder, along with a tiny FPN-like convolutional head/decoder, closes the gap entirely to various SoTA image segmentation models.

We propose various sub-families of FFNets, comprised entirely of well supported convolutional blocks, and show their efficacy on desktop and mobile hardware. The networks provided are for various input/output ratios, and would be useful for a wide variety of tasks beyond just image segmentation.


We argue that such simple architectures should be the go-to baselines for various computer vision tasks, and even where they might lack in accuracy against more complex models, the simplicity of optimizing and deploying them makes these a worthy consideration. See the [paper](https://arxiv.org/abs/2206.08236) for details of the models, and extensive comparisons.

## License 
This software may be subject to U.S. and international export, re-export, or transfer (“export”) laws.  Diversion contrary to U.S. and international law is strictly prohibited.
See the included [license](LICENSE).

## Citing
If you use the models or the weight definitions, please cite the following publication:

```
 @inproceedings{mehta2022simple,
  title={Simple and Efficient Architectures for Semantic Segmentation},
  author={Mehta, Dushyant and Skliar, Andrii and Ben Yahia, Haitam and Borse, Shubhankar and Porikli, Fatih and Habibian, Amirhossein and Blankevoort, Tijmen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2628--2636},
  year={2022}
}
```


## Setup
### Docker Business
```
docker build --no-cache -t ffnet_release docker/
docker run --ipc=host --gpus all --rm -it -v "<path_of_imagenet>:/workspace/imagenet/" -v "<path_of_cityscapes>:/workspace/cityscapes/" -v "<path_of_model_weights>:/workspace/ffnet_weights/"  -v "<local_path_to_repo>:/workspace/ffnet_release/" -e PYTHONPATH=/workspace/ffnet_release -w /workspace/ffnet_release ffnet_release 
```


### Setting paths
config.py: Paths to imagenet and cityscapes, as well as to the directory with the model weights are here. The default paths are as above in the docker command. In case you choose to map them to a different path, please update the respective paths in config.py

### Notes on Dependencies
For inference and evaluation on ImageNet and Cityscapes, the provided docker container suffices. The only somewhat extraneous dependency is SciPy, which is only being used to create a gaussian kernel in ffnet_blocks.py. You can create a gaussian filtering kernel in another way, or replace the anti-aliased downsampling implemented as gaussian filtering + strided convolution with another operation similar in spirit. YMMV.

### Pre-trained models
We make weights available with supervised ImageNet pretraining, intended for further downstream usage, as well as weights for Cityscapes semantic segmentation.
Use the included script fetch_pre_trained_weights.sh under the model_weights directory, to download the weights.
**Update 14 March 2024** We also release ImageNet Self Supervised Training weights, trained using [PixPro](https://github.com/zdaxie/PixPro), for downstream usage on tasks such as semantic segmentation, instance segmentation, scene depth, object detection etc.


## Usage
This repository provides the model definitions for ImageNet image classification, and Cityscapes semantic segmentation, as well as code for evaluating on the respective datasets.
The model definitions are split across files for deployment scenario based grouping.
We also provide example model definitions for interfacing with training pipelines.

### ImageNet evaluation
``` python scripts/evaluate_imagenet.py --gpu_id 0 --model_name classification_ffnet54S_BBX_mobile ```

### Cityscapes evaluation
``` python scripts/evaluate_cityscapes.py --gpu_id 0 --model_name segmentation_ffnet40S_BBB_mobile_pre_down --fp16 ```

### Inference time evaluation
```  python scripts/evaluate_timing.py --num_iter 200 --gpu_id 0 --model_name segmentation_ffnet122NS_CBB_mobile ```


## Model Documentation

### GPU Models (Large)
Images of 1024x2048 are input to these models, and feature maps of 256x512 are output. These models provide a much better speed-accuracy tradeoff than HRNets, as shown in the paper. The 3-stage models (ffnet122N/74N/46N) also provide a much better speed-accuracy tradeoff than DDNets, FANets, and the models listed under "GPU Models (Small)".
*= 3-stage FFNets

![Comparisons of Large GPU Models](figures/stride_32_gpu_models.png?raw=true "FFNets are extremely competitive with SoTA architectures like HRNets, while being simple.")

| **FFNet   GPU Large** | **Model Name In Repo** | **ImageNet Backbone Model** | **Cityscapes Accuracy** | **FP32 (ms)** | **FP16 (ms)** |
|---|---|---|---|---|---|
| ResNet 101 A-A-A | segmentation_ffnet101_AAA | classification_ffnet101_AAX | 82.1 | 119 | 59 |
| ResNet 50 A-A-A | segmentation_ffnet50_AAA | classification_ffnet50_AAX | 79.6 | 88 | 45 |
|  |  |  |  |  |  |
| ResNet 150 A-A-A | segmentation_ffnet150_AAA | classification_ffnet150_AAX | 84.4 | 152 | 81 |
| ResNet 134 A-A-A | segmentation_ffnet134_AAA | classification_ffnet134_AAX | 84.1 | 135 | 70 |
| ResNet 86 A-A-A | segmentation_ffnet86_AAA | classification_ffnet86_AAX | 83.2 | 105 | 55 |
| ResNet 56 A-A-A | segmentation_ffnet56_AAA | classification_ffnet56_AAX | 82.5 | 82 | 42 |
| ResNet 34 A-A-A | segmentation_ffnet34_AAA | classification_ffnet34_AAX | 81.4 | 67 | 34 |
|  |  |  |  |  |  |
| ResNet 150 A-B-B | segmentation_ffnet150_ABB | classification_ffnet150_AAX | 83.7 | 125 | 71 |
| ResNet 86 A-B-B | segmentation_ffnet86_ABB | classification_ffnet86_AAX | 83.5 | 78 | 45 |
| ResNet 56 A-B-B | segmentation_ffnet56_ABB | classification_ffnet56_AAX | 82.1 | 56 | 32 |
| ResNet 34 A-B-B | segmentation_ffnet34_ABB | classification_ffnet34_AAX | 80.3 | 41 | 25 |
|  |  |  |  |  |  |
| ResNet 150 S B-B-B | segmentation_ffnet150S_BBB | classification_ffnet150S_BBX | 84.1 | 104 | 66 |
| ResNet 86 S B-B-B | segmentation_ffnet86S_BBB | classification_ffnet86S_BBX | 82.6 | 67 | 43 |
|  |  |  |  |  |  |
| ResNet 122 N C-B-B* | segmentation_ffnet122N_CBB | classification_ffnet122N_CBX | 83.7 | 58 | 44 |
| ResNet 74 N C-B-B* | segmentation_ffnet74N_CBB | classification_ffnet74N_CBX | 83 | 42 | 32 |
| ResNet 46 N C-B-B* | segmentation_ffnet46N_CBB | classification_ffnet46N_CBX | 81.9 | 34 | 27 |

### GPU Models (Small)
Images of 1024x2048 are input to these models, and feature maps of 128x256 are output. These models provide a better speed-accuracy tradeoff than DDRNets and FANets, as shown in the paper.

![Comparisons of Small GPU Models](figures/stride_64_gpu_models.png?raw=true "FFNets are extremely competitive with SoTA architectures like DDRNets, while being simple.")

| **FFNet   GPU Small** | **Model Name In Repo** | **ImageNet Backbone Model** | **Cityscapes Accuracy** | **FP32 (ms)** | **FP16 (ms)** |
|---|---|---|---|---|---|
| ResNet 101 A-A-A | segmentation_ffnet101_dAAA | classification_ffnet101_AAX | 80.4 | 36 | 29 |
| ResNet 50 A-A-A | segmentation_ffnet50_dAAA | classification_ffnet50_AAX | 79.4 | 27 | 20 |
|  |  |  |  |  |  |
| ResNet 150 A-A-A | segmentation_ffnet150_dAAA | classification_ffnet150_AAX | 82.3 | 41 | 37 |
| ResNet 134 A-A-A | segmentation_ffnet134_dAAA | classification_ffnet134_AAX | 82 | 38 | 35 |
| ResNet 86 A-A-A | segmentation_ffnet86_dAAA | classification_ffnet86_AAX | 81.4 | 30 | 28 |
| ResNet 56 A-A-A | segmentation_ffnet56_dAAA | classification_ffnet56_AAX | 80.7 | 25 | 22 |
| ResNet 34 A-A-A | segmentation_ffnet34_dAAA | classification_ffnet34_AAX | 79.1 | 21 | 18 |
| ResNet 18 A-A-A | segmentation_ffnet18_dAAA | classification_ffnet18_AAX | 76.5 | 19 | 14 |
|  |  |  |  |  |  |
| ResNet 150 A-A-C | segmentation_ffnet150_dAAC | classification_ffnet150_AAX | 81.9 | 37 | 33 |
| ResNet 86 A-A-C | segmentation_ffnet86_dAAC | classification_ffnet86_AAX | 81.1 | 26 | 26 |
| ResNet 34 A-A-C | segmentation_ffnet34_dAAC | classification_ffnet34_AAX | 79.1 | 17 | 16 |
| ResNet 18 A-A-C | segmentation_ffnet18_dAAC | classification_ffnet18_AAX | 76.4 | 15 | 12 |
|  |  |  |  |  |  |
| ResNet 150 S B-B-B | segmentation_ffnet150S_dBBB | classification_ffnet150S_BBX | 81 | 31 | 34 |
| ResNet 86 S B-B-B | segmentation_ffnet86S_dBBB | classification_ffnet86S_BBX | 81.1 | 23 | 26 |


### Mobile Models
These are designed for on-device usage, but are also efficient on desktop GPUs. In that setting, better mileage can be obtained from these models by using bilinear or another upsampling in place of nearest neighbour upsampling. It is recommended to re-train the imagenet backbone if making such changes to the architecture.
*=  3-stage FFNets



| **FFNet   Mobile** | **Model Name In Repo** | **ImageNet Backbone Model** | **Cityscapes Accuracy** | **Cityscapes Input Size** | **Output Size** |
|---|---|---|---|---|---|
| ResNet 86 S B-B-B | segmentation_ffnet86S_dBBB_mobile | classification_ffnet86S_BBX_mobile | 81.5 | 1024x2048 | 128x256 |
| ResNet 78 S B-B-B | segmentation_ffnet78S_dBBB_mobile | classification_ffnet78S_BBX_mobile | 81.3 | 1024x2048 | 128x256 |
| Resnet 54 S B-B-B | segmentation_ffnet54S_dBBB_mobile | classification_ffnet54S_BBX_mobile | 80.8 | 1024x2048 | 128x256 |
| ResNet 40 S B-B-B | segmentation_ffnet40S_dBBB_mobile | classification_ffnet40S_BBX_mobile | 79.2 | 1024x2048 | 128x256 |
|  |  |  |  |  |  |
| ResNet 150 S B-B-B | segmentation_ffnet150S_BBB_mobile(_pre_down) | classification_ffnet150S_BBX_mobile | 81.6 | 512x1024 | 128x256 |
| ResNet 86 S B-B-B | segmentation_ffnet86S_BBB_mobile(_pre_down) | classification_ffnet86S_BBX_mobile | 80.9 | 512x1024 | 128x256 |
| ResNet 78 S B-B-B | segmentation_ffnet78S_BBB_mobile(_pre_down) | classification_ffnet78S_BBX_mobile | 80.5 | 512x1024 | 128x256 |
| Resnet 54 S B-B-B | segmentation_ffnet54S_BBB_mobile(_pre_down) | classification_ffnet54S_BBX_mobile | 80.2 | 512x1024 | 128x256 |
| ResNet 40 S B-B-B | segmentation_ffnet40S_BBB_mobile(_pre_down) | classification_ffnet40S_BBX_mobile | 79.7 | 512x1024 | 128x256 |
|  |  |  |  |  |  |
| ResNet 150 S B-C-C | segmentation_ffnet150S_BCC_mobile(_pre_down) | classification_ffnet150S_BBX_mobile | 81.0 | 512x1024 | 128x256 |
| ResNet 86 S B-C-C | segmentation_ffnet86S_BCC_mobile(_pre_down) | classification_ffnet86S_BBX_mobile | 81.0 | 512x1024 | 128x256 |
| ResNet 78 S B-C-C | segmentation_ffnet78S_BCC_mobile(_pre_down) | classification_ffnet78S_BBX_mobile | 80.6 | 512x1024 | 128x256 |
| Resnet 54 S B-C-C | segmentation_ffnet54S_BCC_mobile(_pre_down) | classification_ffnet54S_BBX_mobile | 79.9 | 512x1024 | 128x256 |
| ResNet 40 S B-C-C | segmentation_ffnet40S_BCC_mobile(_pre_down) | classification_ffnet40S_BBX_mobile | 78.4 | 512x1024 | 128x256 |
|  |  |  |  |  |  |
| FFNet 122 NS C-B-B* | segmentation_ffnet122NS_CBB_mobile(_pre_down) | classification_ffnet122NS_CBX_mobile | 79.3 | 512x1024 | 128x256 |
| FFNet 74 NS C-B-B* | segmentation_ffnet74NS_CBB_mobile(_pre_down) | classification_ffnet74NS_CBX_mobile | 78.3 | 512x1024 | 128x256 |
| FFNet 46 NS C-B-B* | segmentation_ffnet46NS_CBB_mobile(_pre_down) | classification_ffnet46NS_CBX_mobile | 77.5 | 512x1024 | 128x256 |
|  |  |  |  |  |  |
| FFNet 122 NS C-C-C* | segmentation_ffnet122NS_CCC_mobile(_pre_down) | classification_ffnet122NS_CBX_mobile | 79.2 | 512x1024 | 128x256 |
| FFNet 74 NS C-C-C* | segmentation_ffnet74NS_CCC_mobile(_pre_down) | classification_ffnet74NS_CBX_mobile | 77.8 | 512x1024 | 128x256 |
| FFNet 46 NS C-C-C* | segmentation_ffnet46NS_CCC_mobile(_pre_down) | classification_ffnet46NS_CBX_mobile | 76.7 | 512x1024 | 128x256 |


## Additional Notes 
### ImageNet training hyperparameters
The models are trained using timm
LR (0.048 / (2*384)) * 128 * 8, batch size 128x8 or 192x6, epochs 150
Step LR scheduler, decay_epochs 2, using amp, decay_rate of 0.93, rmsproptf optimizer, opt_eps 1e-3
Warmup LR 1e-6, weight decay 4e-5, dropout rate 0.2, model ema, autoaugment rand-m9-mstd0.5, remode pixel, reprob 0.3
Some imagenet models can be made better initializers for Cityscapes by retraining them with a batch size of 192x6, starting from the first imagenet training run.
FFNet150 is trained with 128x10, for 150 epochs.

### Downstream training
Details of training on the cityscapes dataset are in the paper. The only aspect pertinent to other tasks / datasets may be initializing the **up-head** from scratch when finetuning from ImageNet weights. For Cityscapes we found it to consistently give better performance than when initialized with ImageNet weights.


## Acknowledgements:

This repository adapts code from the following repositories:

* Part of UpBranch in FFNet from: FANet
https://github.com/feinanshan/FANet

* Part of ClassificationHead in FFNet from: HRNet
https://github.com/HRNet/HRNet-Image-Classification

* ResNet model definition from: Torchvision
https://github.com/pytorch/vision/tree/main/torchvision

* Cityscapes evaluation code from: InverseForm
https://github.com/Qualcomm-AI-research/InverseForm

which, in turn, borrows from Hierarchical Multi-Scale Attention(HMS): 
https://github.com/NVIDIA/semantic-segmentation

* Part of ImageNet evaluation, and model inference timing code from: SelecSLS-Pytorch
https://github.com/mehtadushy/SelecSLS-Pytorch

We would like to acknowledge the researchers who made these repositories open-source.
