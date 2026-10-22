import torchvision
import torch
from ultralytics import YOLO
from openvino import convert_model
model = YOLO("D:/python/yolov5-7.0/yolo11n-cls.pt")  # load an official model
ov_model = convert_model(model, input=torch.zeros(1, 3, 224, 224))
