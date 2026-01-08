"""
实时检测模块 - 支持图像和视频检测
使用YOLOv8进行目标检测 + ResNext101进行分类
"""
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os

class GarbageDetector:
    def __init__(self, model_path, class_names, device='cuda', conf_threshold=0.3):
        """
        初始化垃圾检测器
        
        Args:
            model_path: ResNext101分类模型路径
            class_names: 分类类别名称列表
            device: 设备 ('cuda' 或 'cpu')
            conf_threshold: 分类置信度阈值，低于此值不显示框
        """
        self.device = device
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        
        # 加载YOLOv8目标检测模型
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')  # 使用nano版本，速度快
            print("✓ YOLOv8模型加载成功")
        except Exception as e:
            print(f"警告: 无法加载YOLOv8模型: {e}")
            print("将使用全图分类模式（无目标检测）")
            self.yolo_model = None
        
        # 加载ResNext101分类模型
        checkpoint = torch.load(model_path, map_location=device)
        
        from resnext_classification.resnext_model import ResNextClassifier
        self.model = ResNextClassifier(len(class_names), pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print("✓ ResNext101分类模型加载成功")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 加载中文字体
        self.font = self._load_chinese_font()
        
        print(f"✓ 检测器初始化完成! 支持 {len(class_names)} 个类别")
        print(f"✓ 置信度阈值: {conf_threshold:.1%} (低于此值不显示检测框)")
    
    def _load_chinese_font(self, font_size=30):
        """
        加载中文字体
        """
        # 尝试加载Windows系统中文字体
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, font_size)
                except:
                    continue
        
        # 如果都失败，使用默认字体
        print("警告: 无法加载中文字体，将使用默认字体")
        return ImageFont.load_default()
    
    def detect_objects(self, image):
        """
        使用YOLO检测图像中的物体
        
        Returns:
            list: 检测到的物体边界框列表 [(x1, y1, x2, y2, conf), ...]
        """
        if self.yolo_model is None:
            # 如果没有YOLO模型，返回整个图像作为检测区域
            h, w = image.shape[:2]
            margin = min(w, h) // 8
            return [(margin, margin, w - margin, h - margin, 1.0)]
        
        # 使用YOLO检测物体
        results = self.yolo_model(image, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # 只保留置信度较高的检测
                if conf > 0.25:  # YOLO检测阈值
                    detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        
        return detections
    
    def classify_region(self, image, bbox):
        """
        对图像中的指定区域进行分类
        
        Args:
            image: 原始图像 (numpy array)
            bbox: 边界框 (x1, y1, x2, y2)
        
        Returns:
            pred_class: 预测类别
            confidence: 置信度
        """
        x1, y1, x2, y2 = bbox
        
        # 裁剪区域
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return None, 0.0
        
        # 转换为PIL图像
        region_pil = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        
        # 预处理
        input_tensor = self.transform(region_pil).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        pred_class = self.class_names[predicted.item()]
        conf_score = confidence.item()
        
        return pred_class, conf_score
    
    def detect_image(self, image_path, save_path=None):
        """
        检测图像中的垃圾并标注
        只有当检测到垃圾物体且置信度高于阈值时才显示边界框
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图像 {image_path}")
            return None, [], []
        
        # 使用YOLO检测物体
        detections = self.detect_objects(image)
        
        if not detections:
            print("未检测到任何物体")
            if save_path:
                cv2.imwrite(save_path, image)
            return image, [], []
        
        # 转换为PIL图像以支持中文
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        
        detected_objects = []
        
        # 对每个检测到的物体进行分类
        for det in detections:
            x1, y1, x2, y2, yolo_conf = det
            
            # 对检测区域进行分类
            pred_class, class_conf = self.classify_region(image, (x1, y1, x2, y2))
            
            if pred_class is None:
                continue
            
            # 只有当分类置信度高于阈值时才显示
            if class_conf < self.conf_threshold:
                continue
            
            detected_objects.append({
                'bbox': (x1, y1, x2, y2),
                'class': pred_class,
                'confidence': class_conf
            })
            
            # 根据置信度选择颜色
            if class_conf > 0.8:
                box_color = (0, 255, 0)  # 绿色
            elif class_conf > 0.6:
                box_color = (255, 165, 0)  # 橙色
            else:
                box_color = (255, 0, 0)  # 红色
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=4)
            
            # 准备标签
            label = f"{pred_class}\n{class_conf:.1%}"
            
            # 计算文本大小
            bbox = draw.textbbox((0, 0), label, font=self.font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 文本位置（边界框上方）
            text_x = x1 + 5
            text_y = max(5, y1 - text_height - 15)
            
            # 绘制文本背景
            background_coords = [
                text_x - 5,
                text_y - 5,
                text_x + text_width + 5,
                text_y + text_height + 5
            ]
            draw.rectangle(background_coords, fill=(0, 0, 0, 200))
            
            # 绘制文本
            draw.text((text_x, text_y), label, font=self.font, fill=(255, 255, 255))
        
        # 转换回OpenCV格式
        image_result = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, image_result)
            print(f"检测结果已保存到 {save_path}")
            print(f"检测到 {len(detected_objects)} 个垃圾物体")
            for obj in detected_objects:
                print(f"  - {obj['class']}: {obj['confidence']:.2%}")
        
        return image_result, detected_objects, detections
    
    def _draw_detections_on_frame(self, frame, detections):
        """
        在视频帧上绘制检测结果（支持中文）
        只有检测到垃圾物体且置信度高于阈值时才绘制
        
        Args:
            frame: 视频帧
            detections: 检测结果列表
        
        Returns:
            绘制后的帧
        """
        if not detections:
            return frame
        
        # 转换为PIL图像
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        # 绘制每个检测到的物体
        for obj in detections:
            x1, y1, x2, y2 = obj['bbox']
            pred_class = obj['class']
            confidence = obj['confidence']
            
            # 根据置信度选择颜色
            if confidence > 0.8:
                box_color = (0, 255, 0)
            elif confidence > 0.6:
                box_color = (255, 165, 0)
            else:
                box_color = (255, 0, 0)
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            
            # 准备标签
            label = f"{pred_class}\n{confidence:.1%}"
            
            # 计算文本大小
            bbox = draw.textbbox((0, 0), label, font=self.font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 文本位置
            text_x = x1 + 5
            text_y = max(5, y1 - text_height - 10)
            
            # 绘制文本背景
            background_coords = [
                text_x - 5,
                text_y - 5,
                text_x + text_width + 5,
                text_y + text_height + 5
            ]
            draw.rectangle(background_coords, fill=(0, 0, 0, 200))
            
            # 绘制文本
            draw.text((text_x, text_y), label, font=self.font, fill=(255, 255, 255))
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    
    def detect_video(self, video_path, save_path=None, show_window=True):
        """
        检测视频中的垃圾
        只有检测到垃圾物体且置信度高于阈值时才显示边界框
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频信息: {width}x{height} @ {fps}fps")
        
        # 视频写入器
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detected_objects = []
        
        print("开始处理视频... (按 'q' 退出)")
        print(f"置信度阈值: {self.conf_threshold:.1%} (低于此值不显示)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 每5帧检测一次（提高速度）
            if frame_count % 5 == 0:
                # 检测物体
                yolo_detections = self.detect_objects(frame)
                detected_objects = []
                
                # 对每个检测到的物体进行分类
                for det in yolo_detections:
                    x1, y1, x2, y2, yolo_conf = det
                    pred_class, class_conf = self.classify_region(frame, (x1, y1, x2, y2))
                    
                    # 只有置信度高于阈值才保留
                    if pred_class and class_conf >= self.conf_threshold:
                        detected_objects.append({
                            'bbox': (x1, y1, x2, y2),
                            'class': pred_class,
                            'confidence': class_conf
                        })
            
            # 绘制检测结果
            frame = self._draw_detections_on_frame(frame, detected_objects)
            
            if writer:
                writer.write(frame)
            
            if show_window:
                cv2.imshow('Garbage Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"视频检测完成! 共处理 {frame_count} 帧")
        if save_path:
            print(f"结果已保存到 {save_path}")
    
    def detect_camera(self, camera_id=0):
        """
        使用摄像头实时检测
        只有检测到垃圾物体且置信度高于阈值时才显示边界框
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return
        
        print("摄像头实时检测启动...")
        print(f"置信度阈值: {self.conf_threshold:.1%} (低于此值不显示)")
        print("按 'q' 退出")
        
        detected_objects = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            frame_count += 1
            
            # 每3帧检测一次（提高实时性）
            if frame_count % 3 == 0:
                # 检测物体
                yolo_detections = self.detect_objects(frame)
                detected_objects = []
                
                # 对每个检测到的物体进行分类
                for det in yolo_detections:
                    x1, y1, x2, y2, yolo_conf = det
                    pred_class, class_conf = self.classify_region(frame, (x1, y1, x2, y2))
                    
                    # 只有置信度高于阈值才保留
                    if pred_class and class_conf >= self.conf_threshold:
                        detected_objects.append({
                            'bbox': (x1, y1, x2, y2),
                            'class': pred_class,
                            'confidence': class_conf
                        })
            
            # 绘制检测结果
            frame = self._draw_detections_on_frame(frame, detected_objects)
            
            cv2.imshow('Real-time Garbage Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头检测已停止")
