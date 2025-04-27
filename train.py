# -*- coding: utf-8 -*-
# @Time    : 2025/4/27 09:20:57
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : train.py
# @Desc    :

import os
from ultralytics import YOLO

model_path = './model/yolo8n.pt'

model = YOLO(model_path) if os.path.exists(model_path) else YOLO('yolo8n.yaml')
