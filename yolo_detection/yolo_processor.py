"""
YOLOv8目标检测和数据预处理模块
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

class YOLOProcessor:
    def __init__(self, model_path='yolov8s.pt', conf_threshold=0.25):
        """
        初始化YOLO处理器
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def detect_objects(self, image_path):
        """
        检测图像中的垃圾目标
        """
        results = self.model(image_path, conf=self.conf_threshold)
        return results
    
    def visualize_detection(self, image_path, save_path=None):
        """
        可视化检测结果
        """
        results = self.detect_objects(image_path)
        
        # 绘制检测结果
        annotated_img = results[0].plot()
        
        if save_path:
            cv2.imwrite(save_path, annotated_img)
        
        return annotated_img, results
    
    def crop_detected_objects(self, image_path, output_dir, min_size=50):
        """
        裁剪检测到的垃圾目标
        """
        results = self.detect_objects(image_path)
        image = cv2.imread(image_path)
        
        cropped_objects = []
        boxes = results[0].boxes
        
        for idx, box in enumerate(boxes):
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 过滤太小的目标
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                continue
            
            # 裁剪目标
            cropped = image[y1:y2, x1:x2]
            
            # 保存裁剪的目标
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                img_name = Path(image_path).stem
                crop_path = os.path.join(output_dir, f"{img_name}_obj_{idx}.jpg")
                cv2.imwrite(crop_path, cropped)
                cropped_objects.append(crop_path)
        
        return cropped_objects
