# -*- coding: utf-8 -*-
# @Time    : 2025/4/30 16:11:00
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : quantized.py
# @Desc    :


from ultralytics import YOLO
import multiprocessing
from onnxruntime.quantization import quantize_dynamic, QuantType

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model_path = 'best.pt'

    model = YOLO(model_path)
    model.export(format="onnx", half=True)
    onnx_model_path = 'best.onnx'
    quantized_model_path = 'quantized_best.onnx'
    quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QUInt8)
