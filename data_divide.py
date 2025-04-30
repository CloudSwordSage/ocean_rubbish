# -*- coding: utf-8 -*-
# @Time    : 2025/4/27 10:10:26
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : data_divide.py
# @Desc    :


import os
import json
import shutil
import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

if os.path.exists("./datasets"):
    shutil.rmtree("./datasets")

os.makedirs("./datasets/images/train", exist_ok=True)
os.makedirs("./datasets/images/val", exist_ok=True)
os.makedirs("./datasets/labels/train", exist_ok=True)
os.makedirs("./datasets/labels/val", exist_ok=True)

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.3),
    A.Blur(blur_limit=3, p=0.1),
    A.CoarseDropout(num_holes=8, max_h_size=20, max_w_size=20, fill_value=0, p=0.5),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

classes = []
with open('./datasets_bak/classes.txt', 'r') as f:
    classes.extend(line.strip() for line in f)

def augment_image(image_path, label_path, save_dir, augment_num=2):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(label_path, 'r') as f:
        annotations = [line.strip().split() for line in f]
    bboxes = [[float(x) for x in ann[1:]] for ann in annotations]
    class_labels = [int(ann[0]) for ann in annotations]

    for i in range(augment_num):
        transformed = train_transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )

        new_image = cv2.cvtColor(transformed['image'].permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
        new_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_aug{i}.jpg"
        cv2.imwrite(os.path.join(save_dir, 'images/train', new_filename), new_image)

        with open(os.path.join(save_dir, 'labels/train', new_filename.replace('.jpg', '.txt')), 'w') as f:
            for bbox, cls in zip(transformed['bboxes'], transformed['class_labels']):
                f.write(f"{cls} {' '.join(map(str, bbox))}\n")

images = os.listdir("./datasets_bak/images")
random.shuffle(images)
train_images = images[:int(len(images)*0.8)]
val_images = images[int(len(images)*0.8):]

for image in train_images:
    src_img = os.path.join("./datasets_bak/images", image)
    dst_img = os.path.join("./datasets/images/train", image)
    shutil.copy(src_img, dst_img)

    src_label = os.path.join("./datasets_bak/labels", image.replace(".jpg", ".txt"))
    dst_label = os.path.join("./datasets/labels/train", image.replace(".jpg", ".txt"))
    shutil.copy(src_label, dst_label)

    augment_image(src_img, src_label, "./datasets")

for image in val_images:
    shutil.copy(os.path.join("./datasets_bak/images", image),
                "./datasets/images/val")
    shutil.copy(os.path.join("./datasets_bak/labels", image.replace(".jpg", ".txt")),
                "./datasets/labels/val")

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
