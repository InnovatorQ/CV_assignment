# 🗑️ 垃圾分类系统

基于 YOLOv8 + ResNext101 的智能垃圾检测与分类系统

## 🌟 项目特点

- **智能检测**: 使用YOLOv8进行目标检测，只在检测到垃圾物体时显示边界框
- **精准分类**: 使用ResNext101对检测到的物体进行40类垃圾分类
- **中文支持**: 完美显示中文标签，无乱码
- **实时检测**: 支持图像、视频、摄像头三种检测模式
- **高性能**: GPU加速，实时检测流畅运行

## 📊 系统架构

```
输入 → YOLOv8目标检测 → ResNext101分类 → 条件显示
                ↓                ↓              ↓
            检测物体        垃圾分类      置信度过滤
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

**系统要求**:
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (推荐，用于GPU加速)

### 2. 快速测试

```bash
# 测试图像检测
python demo.py results\detected_img_6.jpg --no-show

# 或使用完整检测功能
python main.py --step 4
```

### 3. 检测模式

运行 `python main.py --step 4` 后选择：
- **1** - 图像检测
- **2** - 视频检测
- **3** - 摄像头实时检测

## 📁 项目结构

```
垃圾分类系统/
├── config.py                 # 配置文件
├── main.py                   # 主程序入口
├── demo.py                   # 快速演示脚本
├── convert_dataset.py        # 数据集转换工具
├── requirements.txt          # 依赖包列表
├── quick_start.bat          # Windows快速启动脚本
│
├── docs/                    # 📚 文档目录
│   ├── README.md           # 文档导航
│   ├── START_HERE.md       # 快速入门
│   ├── 新功能使用指南.md   # 最新功能说明
│   ├── 目标检测功能说明.md # 技术说明
│   ├── USAGE_GUIDE.md      # 完整使用指南
│   ├── DETECTION_GUIDE.md  # 检测功能详解
│   └── EVALUATION_REPORT.md # 评估报告
│
├── resnext_classification/  # ResNext101分类模块
│   ├── resnext_model.py    # 模型定义
│   ├── trainer.py          # 训练器
│   └── evaluator.py        # 评估器
│
├── yolo_detection/          # YOLO检测模块
│   ├── yolo_processor.py   # YOLO处理器
│   └── data_preprocessor.py # 数据预处理
│
├── utils/                   # 工具模块
│   └── detector.py         # 检测器（YOLOv8 + ResNext101）
│
├── data/                    # 数据目录
│   ├── processed/          # 处理后的数据
│   │   ├── train/         # 训练集
│   │   └── val/           # 验证集
│   └── conversion_stats.json
│
├── models/                  # 模型目录
│   ├── best_resnext_model.pth  # 最佳模型
│   ├── training_history.json  # 训练历史
│   └── training_curves.png    # 训练曲线
│
└── results/                 # 结果目录
    ├── confusion_matrix.png    # 混淆矩阵
    ├── class_accuracies.png   # 类别准确率
    └── detected_*.jpg         # 检测结果图像
```

## 🎯 核心功能

### 1. 智能目标检测
- 使用YOLOv8检测画面中的物体
- 只在检测到垃圾物体时显示边界框
- 没有垃圾时不显示任何标记

### 2. 精准垃圾分类
- 40个细分类别，4个主类别
- 平均准确率77%
- 支持中文类别名称

### 3. 实时检测
- 图像检测: 100-200ms (GPU)
- 视频检测: 30fps (GPU)
- 摄像头检测: 实时流畅

### 4. 置信度过滤
- 默认阈值: 30%
- 低于阈值的检测不显示
- 可自定义调整阈值

## 📊 支持的垃圾类别

### 4个主类别
- 其他垃圾 (6类)
- 厨余垃圾 (8类)
- 可回收物 (23类)
- 有害垃圾 (3类)

### 40个细分类别
包括但不限于：
- 可回收物: 饮料瓶、纸板箱、易拉罐、塑料瓶等
- 厨余垃圾: 果皮、菜叶、剩饭剩菜等
- 有害垃圾: 干电池、过期药物等
- 其他垃圾: 一次性快餐盒、烟蒂等

## 🎨 检测效果

### 有垃圾物体
```
┌─────────────────────────┐
│ 可回收物_饮料瓶         │ ← 中文标签
│ 85.3%                   │ ← 置信度
│  ┏━━━━━━━━━━━━━━━┓     │
│  ┃               ┃     │ ← 绿色边框
│  ┃  [饮料瓶]     ┃     │
│  ┗━━━━━━━━━━━━━━━┛     │
└─────────────────────────┘
```

### 无垃圾物体
```
┌─────────────────────────┐
│                         │
│    [普通场景]           │ ← 没有任何框
│                         │
└─────────────────────────┘
```

## 📖 文档

详细文档请查看 [docs/](docs/) 目录：

- **[快速入门](docs/START_HERE.md)** - 5分钟上手
- **[使用指南](docs/新功能使用指南.md)** - 完整功能说明
- **[技术文档](docs/目标检测功能说明.md)** - 技术细节
- **[评估报告](docs/EVALUATION_REPORT.md)** - 性能分析

## 🔧 配置说明

### 置信度阈值调整

在 `main.py` 或 `demo.py` 中修改：

```python
detector = GarbageDetector(
    model_path=model_path,
    class_names=class_names,
    device=device,
    conf_threshold=0.3  # 默认30%，可调整
)
```

### 边界框颜色
- 🟢 绿色: 置信度 > 80% (高)
- 🟠 橙色: 置信度 60-80% (中)
- 🔴 红色: 置信度 30-60% (低)

## 💻 系统要求

### 最低配置
- CPU: Intel i5 或同等性能
- 内存: 8GB RAM
- 存储: 5GB 可用空间
- 系统: Windows 10/11, Linux, macOS

### 推荐配置
- CPU: Intel i7 或更高
- GPU: NVIDIA RTX 3060 或更高
- 内存: 16GB RAM
- 存储: 10GB 可用空间
- CUDA: 11.8+

## 🚀 性能指标

### 检测速度
| 设备 | 图像检测 | 视频检测 | 摄像头检测 |
|------|---------|---------|-----------|
| RTX 4060 | 100-200ms | 30fps | 30fps |
| CPU | 500-1000ms | 10-15fps | 10-15fps |

### 分类准确率
- 整体准确率: 77.13%
- 最佳类别: 快递纸袋 (91.30%)
- 平均置信度: 85%+

## 🛠️ 开发指南

### 训练自己的模型

```bash
# 1. 准备数据集
python convert_dataset.py

# 2. 训练模型
python main.py --step 2

# 3. 评估模型
python main.py --step 3

# 4. 开始检测
python main.py --step 4
```

### 批量处理

```python
from utils.detector import GarbageDetector
import config

detector = GarbageDetector(
    model_path='models/best_resnext_model.pth',
    class_names=config.CLASS_NAMES,
    device='cuda',
    conf_threshold=0.3
)

# 批量检测图像
for image_path in image_list:
    detector.detect_image(image_path, save_path)
```

## 📝 更新日志

### v2.0 (2026-01-08)
- ✅ 集成YOLOv8目标检测
- ✅ 实现条件显示（只在检测到垃圾时显示框）
- ✅ 支持多物体同时检测
- ✅ 完善中文显示支持
- ✅ 优化检测性能

### v1.0
- ✅ ResNext101分类模型训练
- ✅ 基础检测功能
- ✅ 模型评估报告

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可证

本项目仅供学习和研究使用。

## 📧 联系方式

如有问题，请查看文档或提交Issue。

---

**项目版本**: v2.0  
**最后更新**: 2026年1月8日  
**开发环境**: Python 3.11 + PyTorch 2.7.0 + CUDA 12.8
