import cv2
import numpy as np
import torch

def main():
    frame = cv2.imread('input.jpg')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model(frame)
    results.print()
