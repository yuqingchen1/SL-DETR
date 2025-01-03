from .LG_dab_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from .models.LG_dab_detr_r50_dc5 import model

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/LG_dab_detr_r50_dc5_50ep"
