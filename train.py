# -*- coding: utf-8 -*-
# @Time    : 2025/4/27 09:20:57
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : train.py
# @Desc    :

import os
from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model_path = 'yolo11s.pt'

    model = YOLO(model_path)

    model.train(data='./datasets/data.yaml', epochs=2)

    model.val()

    model.save('model/yolo11s_last.pt')
