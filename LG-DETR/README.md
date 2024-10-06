# SL-DETR: Universal synchronization loss optimization in DETR-based oriented and rotated object detection
### Yu Qingchen


## Installation 🛠️

Install details can be found in [installation instructions](INSTALL.md) 

## Usage 📖
Train Example
```bash
python tools/train_net.py --config-file  lgdetr/lgdetr_k=2_r50_4scale_12ep.py --num-gpus 8
```
Evaluation Example
```bash
python tools/train_net.py --config-file  lgdetr/lgdetr_k=2_r50_4scale_12ep.py --num-gpus 8 --eval train.init_checkpoint=/path/to/checkpoint
```
## Model Zoo 🦁
\* represents using  a modified IA-BCE loss that absorbs focal loss term.

| Model            | AP   | AP50 | AP75 | APs  | APm  | APl  | weights                                         |
|------------------|------|------|------|------|------|------|-------------------------------------------------|
| lgdetr-R50-12ep* | 51.1 | 68.1 | 55.6 | 33.7 | 57.7 | 66.3 | https://pan.baidu.com/s/18ZVHOKEd1hipvvPvmrlDQA |
| lgdetr-R50-24ep* | 52.1 | 68.8 | 56.8 | 36.0 | 55.7 | 67.2 | https://pan.baidu.com/s/18ZVHOKEd1hipvvPvmrlDQA |

## Acknowlegements 🙏

Our code is based on [detrex](https://github.com/IDEA-Research/detrex) and [detectron2](https://github.com/facebookresearch/detectron2).

## Citation

If you are interested in our work and use our method in your research, please cite


