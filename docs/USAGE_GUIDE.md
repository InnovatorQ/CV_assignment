# 使用指南 - 垃圾分类系统

## 快速开始

### 方法1: 使用批处理脚本（推荐Windows用户）

双击运行 `quick_start.bat`，按照菜单提示操作：

```
1. 安装依赖包
2. 转换数据集格式 (YOLO转分类)
3. 数据预处理 (步骤1)
4. 训练模型 (步骤2)
5. 评估模型 (步骤3)
6. 实时检测 (步骤4)
7. 完整流程 (转换+步骤1-3)
8. 退出
```

### 方法2: 命令行操作

#### 步骤0: 安装依赖
```bash
C:\Users\Q\miniconda3\envs\pytorch\python.exe -m pip install -r requirements.txt
```

#### 步骤1: 测试环境
```bash
C:\Users\Q\miniconda3\envs\pytorch\python.exe test_environment.py
```

#### 步骤2: 转换数据集格式
```bash
C:\Users\Q\miniconda3\envs\pytorch\python.exe convert_dataset.py
```

这一步会将YOLO格式的数据集（E:\garbage_classify）转换为分类格式，按40个类别组织。

#### 步骤3: 数据预处理和划分
```bash
C:\Users\Q\miniconda3\envs\pytorch\python.exe main.py --step 1
```

这一步会：
- 将数据集划分为训练集（80%）和验证集（20%）
- 生成数据分布统计图
- 保存数据集信息

#### 步骤4: 训练ResNext101模型
```bash
C:\Users\Q\miniconda3\envs\pytorch\python.exe main.py --step 2
```

这一步会：
- 加载预训练的ResNext101模型
- 在垃圾分类数据集上训练
- 自动保存最佳模型
- 生成训练曲线

**注意**: 训练时间取决于数据集大小和硬件配置，建议使用GPU。

#### 步骤5: 评估模型
```bash
C:\Users\Q\miniconda3\envs\pytorch\python.exe main.py --step 3
```

这一步会：
- 在验证集上评估模型
- 生成详细的分类报告
- 绘制混淆矩阵
- 分析各类别准确率

#### 步骤6: 实时检测
```bash
C:\Users\Q\miniconda3\envs\pytorch\python.exe main.py --step 4
```

支持三种检测模式：
1. 图像检测
2. 视频检测
3. 摄像头实时检测

### 方法3: 一键运行完整流程
```bash
C:\Users\Q\miniconda3\envs\pytorch\python.exe convert_dataset.py
C:\Users\Q\miniconda3\envs\pytorch\python.exe main.py --step all
```

## 数据集说明

### 原始数据集
- 路径: `E:\garbage_classify`
- 格式: YOLO格式（图像+标注文件）
- 类别: 40个细分类别，4个主类别

### 类别结构
```
其他垃圾 (6类):
  - 一次性快餐盒、污损塑料、烟蒂、牙签、破碎花盆及碟碗、竹筷

厨余垃圾 (8类):
  - 剩饭剩菜、大骨头、水果果皮、水果果肉、茶叶渣、菜叶菜根、蛋壳、鱼骨

可回收物 (23类):
  - 充电宝、包、化妆品瓶、塑料玩具、塑料碗盆、塑料衣架、快递纸袋、插头电线
  - 旧衣服、易拉罐、枕头、毛绒玩具、洗发水瓶、玻璃杯、皮鞋、砧板
  - 纸板箱、调料瓶、酒瓶、金属食品罐、锅、食用油桶、饮料瓶

有害垃圾 (3类):
  - 干电池、软膏、过期药物
```

### 转换后的数据集
- 路径: `data/raw/`
- 格式: 分类格式（按类别组织的文件夹）
- 结构:
```
data/raw/
├── 其他垃圾/一次性快餐盒/
├── 其他垃圾/污损塑料/
├── ...
├── 厨余垃圾/剩饭剩菜/
├── ...
├── 可回收物/充电宝/
├── ...
└── 有害垃圾/干电池/
    └── ...
```

## 输出文件说明

### 1. 数据转换阶段
- `data/conversion_stats.json`: 转换统计信息
- `data/raw/`: 转换后的分类格式数据集

### 2. 数据预处理阶段
- `data/processed/train/`: 训练集（80%）
- `data/processed/val/`: 验证集（20%）
- `data/processed/dataset_stats.json`: 数据集统计
- `data/processed/dataset_distribution.png`: 数据分布图

### 3. 模型训练阶段
- `models/best_resnext_model.pth`: 最佳模型权重
- `models/training_history.json`: 训练历史
- `models/training_curves.png`: 训练曲线图

### 4. 模型评估阶段
- `results/classification_report.txt`: 详细分类报告
- `results/confusion_matrix.png`: 混淆矩阵
- `results/class_accuracies.png`: 各类别准确率

### 5. 检测阶段
- `results/detected_*.jpg`: 检测结果图像
- `results/detected_*.mp4`: 检测结果视频

## 常见问题

### Q1: 转换数据集时出错
**A**: 检查以下几点：
- 确认数据集路径正确: `E:\garbage_classify`
- 确认存在 `train_data` 文件夹
- 确认存在 `garbage_classify_rule.json` 文件

### Q2: 训练时显存不足
**A**: 修改 `config.py` 中的 `BATCH_SIZE`，从32改为16或8

### Q3: 训练速度很慢
**A**: 
- 确认使用GPU训练（检查CUDA是否可用）
- 减小图像尺寸（修改 `IMG_SIZE`）
- 减少训练轮数（修改 `NUM_EPOCHS`）

### Q4: 准确率不理想
**A**: 
- 增加训练轮数（50 → 100）
- 调整学习率（0.001 → 0.0001）
- 检查数据集质量
- 增加数据增强

### Q5: 如何只训练主类别（4类）而不是细分类别（40类）
**A**: 需要修改代码，将细分类别映射到主类别。可以创建一个新的配置选项。

## 性能优化建议

### 1. 硬件优化
- 使用GPU训练（NVIDIA显卡 + CUDA）
- 增加内存（建议16GB+）
- 使用SSD存储数据集

### 2. 模型优化
- 使用混合精度训练（AMP）
- 梯度累积（小显存）
- 模型剪枝和量化

### 3. 数据优化
- 数据预加载和缓存
- 多进程数据加载
- 图像预处理优化

## 高级用法

### 1. 自定义训练参数
编辑 `config.py`:
```python
BATCH_SIZE = 16        # 批次大小
NUM_EPOCHS = 100       # 训练轮数
LEARNING_RATE = 0.0001 # 学习率
IMG_SIZE = 256         # 图像大小
```

### 2. 使用预训练模型继续训练
```python
# 在 main.py 中修改
checkpoint = torch.load('models/best_resnext_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 3. 导出模型为ONNX格式
```python
import torch
from resnext_classification.resnext_model import ResNextClassifier

model = ResNextClassifier(num_classes=40)
checkpoint = torch.load('models/best_resnext_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "garbage_classifier.onnx")
```

### 4. 批量检测图像
```python
from utils.detector import GarbageDetector
import os

detector = GarbageDetector(
    model_path='models/best_resnext_model.pth',
    class_names=config.CLASS_NAMES
)

image_dir = 'path/to/images'
for img_file in os.listdir(image_dir):
    if img_file.endswith(('.jpg', '.png')):
        img_path = os.path.join(image_dir, img_file)
        pred_class, confidence = detector.predict_image(img_path)
        print(f"{img_file}: {pred_class} ({confidence:.2%})")
```

## 项目结构
```
E:\assignment_CV\
├── config.py                      # 配置文件
├── convert_dataset.py             # 数据集转换脚本
├── main.py                        # 主程序
├── demo.py                        # 演示脚本
├── test_environment.py            # 环境测试
├── quick_start.bat               # 快速启动脚本
├── requirements.txt               # 依赖包
├── README.md                     # 项目说明
├── USAGE_GUIDE.md                # 使用指南（本文件）
├── DATA_PREPARATION.md           # 数据准备指南
├── PROJECT_OVERVIEW.md           # 项目总览
│
├── yolo_detection/               # YOLOv8模块
├── resnext_classification/       # ResNext101模块
├── utils/                        # 工具模块
│
├── data/                         # 数据目录
│   ├── raw/                     # 转换后的数据
│   └── processed/               # 处理后的数据
│
├── models/                       # 模型保存目录
└── results/                      # 结果输出目录
```

## 下一步

完成基础训练后，可以考虑：
1. 模型优化和调参
2. 部署为Web服务
3. 移动端应用开发
4. 实际场景测试
5. 持续改进和迭代

## 技术支持

如有问题，请查看：
- README.md: 基础使用说明
- PROJECT_OVERVIEW.md: 项目技术细节
- DATA_PREPARATION.md: 数据准备详细说明

或提交Issue反馈问题。
