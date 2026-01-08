"""
垃圾分类系统主程序
"""
import os
import sys
import torch
import argparse
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from yolo_detection.data_preprocessor import DataPreprocessor
from resnext_classification.resnext_model import ResNextClassifier
from resnext_classification.trainer import ResNextTrainer
from resnext_classification.evaluator import ModelEvaluator
from utils.detector import GarbageDetector

def step1_preprocess_data():
    """
    步骤1: 使用YOLOv8预处理数据集
    """
    print("=" * 60)
    print("步骤1: 数据预处理和划分")
    print("=" * 60)
    
    # 创建数据预处理器
    preprocessor = DataPreprocessor(
        raw_data_dir=config.RAW_DATA_DIR,
        output_dir=config.PROCESSED_DATA_DIR,
        train_split=config.TRAIN_SPLIT
    )
    
    # 划分数据集
    stats = preprocessor.split_dataset(random_seed=config.RANDOM_SEED)
    
    # 分析数据集
    preprocessor.analyze_dataset()
    
    print("\n数据预处理完成!")
    return stats

def step2_train_resnext():
    """
    步骤2: 训练ResNext101模型
    """
    print("\n" + "=" * 60)
    print("步骤2: 训练ResNext101分类模型")
    print("=" * 60)
    
    # 检查设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = ResNextClassifier(
        num_classes=config.NUM_CLASSES,
        pretrained=config.RESNEXT_PRETRAINED
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = ResNextTrainer(
        model=model,
        train_dir=config.TRAIN_DIR,
        val_dir=config.VAL_DIR,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        device=device
    )
    
    # 训练模型
    history = trainer.train(
        num_epochs=config.NUM_EPOCHS,
        save_dir=config.MODELS_DIR
    )
    
    print("\n模型训练完成!")
    return history

def step3_evaluate_model():
    """
    步骤3: 评估模型性能
    """
    print("\n" + "=" * 60)
    print("步骤3: 模型评估")
    print("=" * 60)
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    # 加载最佳模型
    model_path = os.path.join(config.MODELS_DIR, 'best_resnext_model.pth')
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 重建模型
    model = ResNextClassifier(
        num_classes=config.NUM_CLASSES,
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建验证数据加载器
    from torch.utils.data import DataLoader
    from torchvision import transforms, datasets
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(config.VAL_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 创建评估器
    evaluator = ModelEvaluator(
        model=model,
        val_loader=val_loader,
        class_names=val_dataset.classes,
        device=device
    )
    
    # 完整评估
    results = evaluator.full_evaluation(save_dir=config.RESULTS_DIR)
    
    print("\n模型评估完成!")
    return results

def step4_detect():
    """
    步骤4: 实时检测
    """
    print("\n" + "=" * 60)
    print("步骤4: 垃圾检测")
    print("=" * 60)
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(config.MODELS_DIR, 'best_resnext_model.pth')
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return
    
    # 加载类别名称
    from torchvision import datasets
    val_dataset = datasets.ImageFolder(config.VAL_DIR)
    class_names = val_dataset.classes
    
    # 创建检测器
    detector = GarbageDetector(
        model_path=model_path,
        class_names=class_names,
        device=device,
        conf_threshold=0.3  # 置信度阈值，可调整
    )
    
    while True:
        print("\n选择检测模式:")
        print("1. 检测图像")
        print("2. 检测视频")
        print("3. 摄像头实时检测")
        print("4. 退出系统")
        
        choice = input("\n请选择 (1-4): ").strip()
        
        if choice == '1':
            image_path = input("请输入图像路径: ").strip()
            if os.path.exists(image_path):
                save_path = os.path.join(config.RESULTS_DIR, 'detected_' + os.path.basename(image_path))
                detector.detect_image(image_path, save_path)
            else:
                print(f"文件不存在: {image_path}")
        
        elif choice == '2':
            video_path = input("请输入视频路径: ").strip()
            if os.path.exists(video_path):
                save_path = os.path.join(config.RESULTS_DIR, 'detected_' + os.path.basename(video_path))
                detector.detect_video(video_path, save_path)
            else:
                print(f"文件不存在: {video_path}")
        
        elif choice == '3':
            camera_id = input("请输入摄像头ID (默认0): ").strip()
            camera_id = int(camera_id) if camera_id else 0
            detector.detect_camera(camera_id)
        
        elif choice == '4':
            break
        
        else:
            print("无效选择，请重试")

def main():
    """
    主函数
    """
    # 创建必要的目录
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    parser = argparse.ArgumentParser(description='垃圾分类系统')
    parser.add_argument('--step', type=str, choices=['all', '1', '2', '3', '4'],
                       default='all', help='执行步骤: all(全部), 1(预处理), 2(训练), 3(评估), 4(检测)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("垃圾分类系统 - 基于ResNext101")
    print("=" * 60)
    
    if args.step == 'all' or args.step == '1':
        step1_preprocess_data()
    
    if args.step == 'all' or args.step == '2':
        step2_train_resnext()
    
    if args.step == 'all' or args.step == '3':
        step3_evaluate_model()
    
    if args.step == '4':
        step4_detect()
    
    print("\n" + "=" * 60)
    print("程序执行完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()
