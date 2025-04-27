# -*- coding: utf-8 -*-
# @Time    : 2025/4/27 10:10:26
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : data_divide.py
# @Desc    :


import os
import json
import shutil
import random

if os.path.exists("./datasets"):
    shutil.rmtree("./datasets")

os.makedirs("./datasets/images/train", exist_ok=True)
os.makedirs("./datasets/images/val", exist_ok=True)
os.makedirs("./datasets/labels/train", exist_ok=True)
os.makedirs("./datasets/labels/val", exist_ok=True)

images = os.listdir("./datasets_bak/images")

random.shuffle(images)

train_images = images[:int(len(images) * 0.8)]
val_images = images[int(len(images) * 0.8):]

for image in train_images:
    shutil.copy(os.path.join("./datasets_bak/images", image), "./datasets/images/train")
    shutil.copy(os.path.join("./datasets_bak/labels", image.replace(".jpg", ".txt")), "./datasets/labels/train")

for image in val_images:
    shutil.copy(os.path.join("./datasets_bak/images", image), "./datasets/images/val")
    shutil.copy(os.path.join("./datasets_bak/labels", image.replace(".jpg", ".txt")), "./datasets/labels/val")

shutil.copy('./datasets_bak/classes.txt', './datasets/')

classes = []
with open('./datasets_bak/classes.txt', 'r') as f:
    classes.extend(line.strip() for line in f)

root = os.getcwd()

with open('./datasets/data.yaml', 'w') as f:
    f.write(f"train: {os.path.join(root, 'datasets/images/train')}\n")
    f.write(f"val: {os.path.join(root, 'datasets/images/val')}\n")
    f.write(f"nc: {len(classes)}\n")
    f.write(f"names: {json.dumps(classes)}\n")
