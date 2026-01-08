"""
快速演示脚本 - 测试单张图像分类
"""
import os
import sys
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.detector import GarbageDetector
import config

def demo_single_image(image_path, model_path=None, show=True):
    """
    演示单张图像分类
    """
    if model_path is None:
        model_path = os.path.join(config.MODELS_DIR, 'best_resnext_model.pth')
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先训练模型: python main.py --step 2")
        return
    
    if not os.path.exists(image_path):
        print(f"错误: 找不到图像文件 {image_path}")
        return
    
    # 检测设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载类别名称
    checkpoint = torch.load(model_path, map_location=device)
    if 'class_to_idx' in checkpoint:
        idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        class_names = config.CLASS_NAMES
    
    print(f"类别: {class_names}")
    
    # 创建检测器
    detector = GarbageDetector(
        model_path=model_path,
        class_names=class_names,
        device=device,
        conf_threshold=0.3  # 置信度阈值，可调整
    )
    
    # 预测
    print(f"\n正在检测图像: {image_path}")
    image_result, detected_objects, all_detections = detector.detect_image(image_path)
    
    if not detected_objects:
        print("\n未检测到垃圾物体（或置信度低于阈值）")
        return None, 0.0
    
    print(f"\n检测结果:")
    for i, obj in enumerate(detected_objects, 1):
        print(f"  {i}. {obj['class']}: {obj['confidence']:.2%}")
    
    # 可视化
    if show and detected_objects:
        image = Image.open(image_path)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis('off')
        
        title = f"检测到 {len(detected_objects)} 个垃圾物体\n"
        for obj in detected_objects[:3]:  # 最多显示3个
            title += f"{obj['class']} ({obj['confidence']:.1%})  "
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    # 保存结果
    save_path = os.path.join(config.RESULTS_DIR, 'demo_' + os.path.basename(image_path))
    detector.detect_image(image_path, save_path)
    
    if detected_objects:
        return detected_objects[0]['class'], detected_objects[0]['confidence']
    return None, 0.0

def main():
    parser = argparse.ArgumentParser(description='垃圾分类演示')
    parser.add_argument('image', type=str, help='图像路径')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--no-show', action='store_true', help='不显示图像')
    
    args = parser.parse_args()
    
    demo_single_image(args.image, args.model, not args.no_show)

if __name__ == '__main__':
    main()
