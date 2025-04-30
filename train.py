import os
from ultralytics import YOLO
import multiprocessing
import torch

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.cuda.empty_cache()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model_path = 'yolo11s.pt'

    model = YOLO(model_path)

    # model.train(data='./datasets/data.yaml', epochs=2)
    model.train(
        data='./datasets/data.yaml',
        epochs=400,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        mixup=0.0,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005
    )

    torch.cuda.empty_cache()

    torch.cuda.empty_cache()

    model.val()

    model.save('model/yolo11s_last.pt')
