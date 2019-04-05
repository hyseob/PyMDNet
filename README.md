# py-MDNet

by [Hyeonseob Nam](https://kr.linkedin.com/in/hyeonseob-nam/) and [Bohyung Han](http://cvlab.postech.ac.kr/~bhhan/) at POSTECH

## Introduction
PyTorch implementation of MDNet tracker, which runs at 4.5fps with a single thread and a single GPU (1080Ti)
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
- pyTorch 1.0.0+
- for GPU support: a GPU with ~0.3G memory

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
 - Download [VOT](http://www.votchallenge.net/) datasets into "datasets/vot201x"
``` bash
 python pretrain/prepro_vot.py
 python pretrain/train_mdnet.py
```
