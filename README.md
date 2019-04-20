# py-MDNet

by [Hyeonseob Nam](https://kr.linkedin.com/in/hyeonseob-nam/) and [Bohyung Han](http://cvlab.postech.ac.kr/~bhhan/) at POSTECH

**Update (April, 2019)**
- Migration to python 3.6 & pyTorch 1.0
- Efficiency improvement (~5fps)
- ImagNet-VID pretraining
- Code refactoring

## Results
<img src="./figs/tb100-precision.png" width="400"> <img src="./figs/tb100-success.png" width="400">
<img src="./figs/tb50-precision.png" width="400"> <img src="./figs/tb50-success.png" width="400">
<img src="./figs/otb2013-precision.png" width="400"> <img src="./figs/otb2013-success.png" width="400">

## Introduction
PyTorch implementation of MDNet, which runs at ~5fps with a single CPU core and a single GPU (GTX 1080 Ti).
#### [[Project]](http://cvlab.postech.ac.kr/research/mdnet/) [[Paper]](https://arxiv.org/abs/1510.07945) [[Matlab code]](https://github.com/HyeonseobNam/MDNet)

If you're using this code for your research, please cite:

	@InProceedings{nam2016mdnet,
	author = {Nam, Hyeonseob and Han, Bohyung},
	title = {Learning Multi-Domain Convolutional Neural Networks for Visual Tracking},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2016}
	}
 
## Prerequisites
- python 3.6+
- opencv 3.0+
- [PyTorch 1.0+](http://pytorch.org/) and its dependencies 
- for GPU support: a GPU with ~3G memory

## Usage

### Tracking
```bash
 python tracking/run_tracker.py -s DragonBaby [-d (display fig)] [-f (save fig)]
```
 - You can provide a sequence configuration in two ways (see tracking/gen_config.py):
   - ```python tracking/run_tracker.py -s [seq name]```
   - ```python tracking/run_tracker.py -j [json path]```
 
### Pretraining
 - Download [VGG-M](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) (matconvnet model) and save as "models/imagenet-vgg-m.mat"
 - Pretraining on VOT (exclude OTB)
   - Download [VOT](http://www.votchallenge.net/) datasets into "datasets/VOT/vot201x"
    ``` bash
     python pretrain/prepro_vot.py
     python pretrain/train_mdnet.py -d vot
    ```
 - Pretraining on ImageNet-VID
   - Download ImageNet-VID dataset into "datasets/ILSVRC"
    ``` bash
     python pretrain/prepro_imagenet.py
     python pretrain/train_mdnet.py -d imagenet
    ```

## Third-party re-implementations
- Tensorflow, by @AlexQie: [code](https://github.com/AlexQie/MDNet)
- Tensorflow, by @zhyj3038: [code](https://github.com/zhyj3038/PyMDNet)
- MatconvNet (optimized), by @ZjjConan: [code](https://github.com/ZjjConan/Optimized-MDNet)
