# -*- coding: utf-8 -*-
# @Time    : 2025/4/30 14:29:48
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : onnx_model.py
# @Desc    :


import math
import time
import cv2
import numpy as np
import onnxruntime
from ultralytics import YOLO

with open('./classes.txt', 'r') as f:
    classes = [line.strip() for line in f]

rng = np.random.default_rng(114514)
colors = rng.uniform(0, 255, size=(len(classes), 3)).astype("uint8")

def nms(boxes, scores, threshold):
    if len(boxes) == 0:
        return []

    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []

    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious <= threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def compute_iou(box, boxes):
    x_min = np.maximum(box[0], boxes[:, 0])
    y_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[2], boxes[:, 2])
    y_max = np.minimum(box[3], boxes[:, 3])

    intersection_area = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    return intersection_area / union_area

def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def draw_detections(image, boxes, scores, class_ids, draw_scores=True):
    h, w, _ = image.shape
    size = min(w, h) * 0.0006
    text_thickness = int(min(w, h) * 0.001)

    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id].tolist()

        x1, y1, x2, y2 = box.astype(int)

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 准备标签文本
        label = classes[class_id]
        caption = f'{label} {score:.2%}' if draw_scores else f'{label}'
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, size, text_thickness)
        th = int(th * 1.2)

        # 绘制文字背景
        cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

        # 绘制文字
        cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                   size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return image

class YOLOOnnx:
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # 初始化模型
        self.session = onnxruntime.InferenceSession(path)
        self.get_io_details()

    def __call__(self, image):
        return self.detect(image)

    def get_io_details(self):
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

    def detect(self, image):
        input_tensor = self.preprocess(image)

        outputs = self.session.run(None, {self.input_name: input_tensor})

        return self.postprocess(outputs[0])

    def preprocess(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)

        return input_img[np.newaxis, ...].astype(np.float32)

    def postprocess(self, output):
        predictions = np.squeeze(output).T
        if predictions.size == 0:
            return [], [], []

        num_classes = predictions.shape[1] - 4

        scores = np.max(predictions[:, 4:], axis=1)

        valid_mask = scores > self.conf_threshold
        predictions = predictions[valid_mask]
        scores = scores[valid_mask]

        if scores.size == 0:
            return [], [], []

        boxes = predictions[:, :4]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        boxes = self.rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)

        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height,
                              self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height,
                         self.img_width, self.img_height])
        return boxes

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(15, 9))

    model_path = 'quantized_best.onnx'
    img_path = './datasets/images/train/1035.jpg'

    img = cv2.imread(img_path)

    plt.subplot(1, 2, 1)
    yolo = YOLO('./best.pt')
    s = time.time()
    res = yolo(img)

    boxes = res[0].boxes.xyxy.cpu().numpy()
    scores = res[0].boxes.conf.cpu().numpy()
    class_ids = res[0].boxes.cls.cpu().numpy().astype(int)

    yolo_img = img.copy()
    yolo_img = draw_detections(yolo_img, boxes, scores, class_ids)

    plt.imshow(cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB))
    plt.title(f'YOLOv5-PyTorch, {time.time() - s:.4f}s')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    yoloonnx = YOLOOnnx(model_path, conf_thres=0.3, iou_thres=0.5)
    s = time.time()
    boxes, scores, class_ids = yoloonnx(img)

    onnx_img = img.copy()
    onnx_img = draw_detections(onnx_img, boxes, scores, class_ids)

    plt.imshow(cv2.cvtColor(onnx_img, cv2.COLOR_BGR2RGB))
    plt.title(f'YOLOv5-Quantized ONNX, {time.time() - s:.4f}s')
    plt.axis('off')

    plt.tight_layout()

    plt.show()

