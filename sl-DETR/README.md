# SL-DETR: Universal synchronization loss optimization in DETR-based oriented and rotated object detection


### Yu Qingchen


## Main Results
DOTA-v2.0 (Single-Scale Training and Testing)
| Method |         Backbone         | AP50  |                            Config                          | Download |
| :----: | :----------------------: | :---: | :----------------------------------------------------------: |  :----: |
|SL-DETR| ResNet50 (1024,1024,200) | 61.51 |    [sldetr_r50_dota2](configs/sldetr/sldetr_phc_haus-4scale_r50_2xb2-36e_dotav2.py)      |  [model]() |

DOTA-v1.5 (Single-Scale Training and Testing)
| Method |         Backbone         | AP50  |                            Config                          | Download |
| :----: | :----------------------: | :---: | :----------------------------------------------------------: |  :----: |
|SL-DETR| ResNet50 (1024,1024,200) | 72.57 |    [sldetr_r50_dotav15](configs/sldetr/sldetr_phc_haus-4scale_r50_2xb2-36e_dotav15.py)      |  [model]() |

DOTA-v1.0 (Single-Scale Training and Testing)
| Method |         Backbone         | AP50  |                            Config                          | Download |
| :----: | :----------------------: | :---: | :----------------------------------------------------------: |  :----: |
|SL-DETR| ResNet50 (1024,1024,200) | 79.34 |    [sldetr_r50_dota](configs/sldetr/sldetr_phc_haus-4scale_r50_2xb2-36e_dota.py)      |  [model]() |


## Requirements

### Installation
```bash
# torch>=1.9.1 is required.
pip install openmim mmengine==0.7.3
mim install mmcv==2.0.0
pip install mmdet==3.0.0
pip3 install --no-cache-dir -e ".[optional]"
```
or check the [Dockerfile](docker/Dockerfile).


### Preprare Dataset
Details are described in https://github.com/open-mmlab/mmrotate/blob/main/tools/data/dota/README.md

Specifically, run below code.

```bash
python3 tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_trainval.json

python3 tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_test.json
```


## Training

To train the model(s) in the paper, run this command:

```bash
# DOTA-v2.0 R-50
export CONFIG='configs/sldetr/sldetr_phc_haus-4scale_r50_2xb2-36e_dotav2.py'
bash tools/dist_train.sh $CONFIG 2
```

## Evaluation

To evaluate our models on DOTA, run:

```bash
# example
export CONFIG='configs/sldetr/sldetr_phc_haus-4scale_r50_2xb2-36e_dotav2.py'
export CKPT='work_dirs/sldetr_phc_haus-4scale_r50_2xb2-36e_dotav2/epoch_36.pth'
python3 tools/test.py $CONFIG $CKPT
```
Evaluation is processed in the [official DOTA evaluation server](https://captain-whu.github.io/DOTA/evaluation.html).



