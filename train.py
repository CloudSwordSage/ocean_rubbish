import os
from ultralytics import YOLO
import multiprocessing
import torch

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model_path = 'yolo11s.pt'

    model = YOLO(model_path)

    model.train(data='./datasets/data.yaml', epochs=2)
    torch.cuda.empty_cache()

    model.val()

    model.save('model/yolo11s_last.pt')
