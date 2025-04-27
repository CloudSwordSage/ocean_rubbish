# -*- coding: utf-8 -*-
# @Time    : 2025/4/27 09:55:42
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : data_marge.py
# @Desc    :


import os
import json
import shutil

os.makedirs("./datasets_bak/images")
os.makedirs("./datasets_bak/labels")

for root, dirs, files in os.walk("./data"):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".bmp"):
            shutil.copy(os.path.join(root, file), "./datasets_bak/images")

with open("./data/dataset.json", "r") as f:
    data = json.load(f)

category = {item["id"]: item["name"] for item in data["categories"]}

images = {item["id"]: (item['file_name'], item['height'], item['width']) for item in data["images"]}

for item in data["annotations"]:
    image_id = item["image_id"]
    image_name, height, width = images[image_id]
    image_base_name = os.path.splitext(image_name)[0]
    x, y, w, h = item["bbox"]
    center_x = x + w / 2
    center_y = y + h / 2
    x_norm = center_x / width
    y_norm = center_y / height
    w_norm = w / width
    h_norm = h / height
    class_id = item["category_id"]
    with open(f"./datasets_bak/labels/{image_base_name}.txt", "a") as f:
        f.write(f"{class_id} {x_norm} {y_norm} {w_norm} {h_norm}\n")

with open('./datasets_bak/classes.txt', 'w') as f:
    for value in category.values():
        f.write(f"{value}\n")
