# 📁 项目文件结构

## 目录树

```
垃圾分类系统/
│
├── 📄 README.md                    # 项目主文档
├── 📄 PROJECT_STRUCTURE.md         # 本文件 - 项目结构说明
├── 📄 config.py                    # 配置文件
├── 📄 main.py                      # 主程序入口
├── 📄 demo.py                      # 快速演示脚本
├── 📄 convert_dataset.py           # 数据集转换工具
├── 📄 requirements.txt             # Python依赖包列表
├── 📄 quick_start.bat             # Windows快速启动脚本
├── 📄 .gitignore                  # Git忽略文件配置
├── 📄 yolov8n.pt                  # YOLOv8模型文件
│
├── 📂 docs/                        # 📚 文档目录
│   ├── README.md                  # 文档导航
│   ├── START_HERE.md              # 快速入门指南
│   ├── 新功能使用指南.md          # 最新功能使用说明
│   ├── 目标检测功能说明.md        # 目标检测技术说明
│   ├── USAGE_GUIDE.md             # 完整使用指南
│   ├── DETECTION_GUIDE.md         # 检测功能详细说明
│   └── EVALUATION_REPORT.md       # 模型评估报告
│
├── 📂 resnext_classification/      # ResNext101分类模块
│   ├── __init__.py                # 模块初始化
│   ├── resnext_model.py           # ResNext101模型定义
│   ├── trainer.py                 # 模型训练器
│   └── evaluator.py               # 模型评估器
│
├── 📂 yolo_detection/              # YOLO检测模块
│   ├── __init__.py                # 模块初始化
│   ├── yolo_processor.py          # YOLO处理器
│   └── data_preprocessor.py       # 数据预处理器
│
├── 📂 utils/                       # 工具模块
│   ├── __init__.py                # 模块初始化
│   └── detector.py                # 检测器（YOLOv8 + ResNext101集成）
│
├── 📂 data/                        # 数据目录
│   ├── conversion_stats.json     # 数据转换统计
│   └── processed/                 # 处理后的数据
│       ├── train/                 # 训练集（40个类别文件夹）
│       ├── val/                   # 验证集（40个类别文件夹）
│       ├── category_mapping.txt  # 类别映射表
│       ├── dataset_stats.json    # 数据集统计信息
│       └── dataset_distribution.png # 数据分布图
│
├── 📂 models/                      # 模型目录
│   ├── best_resnext_model.pth    # 最佳ResNext101模型
│   ├── training_history.json     # 训练历史记录
│   └── training_curves.png       # 训练曲线图
│
└── 📂 results/                     # 结果目录
    ├── confusion_matrix.png       # 混淆矩阵
    ├── class_accuracies.png      # 各类别准确率图
    ├── class_accuracies.txt      # 各类别准确率文本
    ├── class_mapping.txt         # 类别映射
    ├── classification_report.txt # 分类报告
    └── detected_*.jpg            # 检测结果图像
```

## 📄 核心文件说明

### 配置与入口

| 文件 | 说明 | 用途 |
|------|------|------|
| `config.py` | 配置文件 | 存储所有系统配置参数 |
| `main.py` | 主程序 | 系统主入口，包含完整流程 |
| `demo.py` | 演示脚本 | 快速测试单张图像检测 |
| `requirements.txt` | 依赖列表 | Python包依赖 |
| `quick_start.bat` | 启动脚本 | Windows快速启动 |

### 数据处理

| 文件 | 说明 | 用途 |
|------|------|------|
| `convert_dataset.py` | 数据转换 | YOLO格式转分类格式 |
| `yolo_detection/data_preprocessor.py` | 数据预处理 | 数据集划分和分析 |

### 模型相关

| 文件 | 说明 | 用途 |
|------|------|------|
| `resnext_classification/resnext_model.py` | 模型定义 | ResNext101架构 |
| `resnext_classification/trainer.py` | 训练器 | 模型训练逻辑 |
| `resnext_classification/evaluator.py` | 评估器 | 模型评估和分析 |
| `utils/detector.py` | 检测器 | YOLOv8 + ResNext101集成 |

### 模型文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `yolov8n.pt` | ~6MB | YOLOv8 nano模型 |
| `models/best_resnext_model.pth` | ~170MB | 训练好的ResNext101模型 |

## 📂 目录详解

### docs/ - 文档目录
存放所有项目文档，包括：
- 快速入门指南
- 使用说明
- 技术文档
- 评估报告

**推荐阅读顺序**:
1. START_HERE.md
2. 新功能使用指南.md
3. USAGE_GUIDE.md

### resnext_classification/ - 分类模块
ResNext101分类模型相关代码：
- `resnext_model.py`: 模型架构定义
- `trainer.py`: 训练逻辑（早停、混合精度等）
- `evaluator.py`: 评估逻辑（混淆矩阵、准确率等）

### yolo_detection/ - 检测模块
YOLO相关代码：
- `yolo_processor.py`: YOLO处理逻辑
- `data_preprocessor.py`: 数据预处理和划分

### utils/ - 工具模块
通用工具：
- `detector.py`: 核心检测器，集成YOLOv8和ResNext101

### data/ - 数据目录
```
data/
├── processed/
│   ├── train/              # 训练集
│   │   ├── 其他垃圾_一次性快餐盒/
│   │   ├── 其他垃圾_污损塑料/
│   │   └── ... (40个类别)
│   └── val/                # 验证集
│       ├── 其他垃圾_一次性快餐盒/
│       └── ... (40个类别)
└── conversion_stats.json   # 转换统计
```

### models/ - 模型目录
存放训练好的模型和训练记录：
- `best_resnext_model.pth`: 最佳模型权重
- `training_history.json`: 训练历史（loss、accuracy等）
- `training_curves.png`: 训练曲线可视化

### results/ - 结果目录
存放检测结果和评估结果：
- 检测结果图像
- 混淆矩阵
- 准确率分析
- 分类报告

## 🔄 数据流

### 训练流程
```
原始数据集 (E:\garbage_classify)
    ↓
convert_dataset.py (转换格式)
    ↓
data/processed/ (训练集 + 验证集)
    ↓
main.py --step 2 (训练)
    ↓
models/best_resnext_model.pth (模型)
```

### 检测流程
```
输入图像/视频
    ↓
utils/detector.py
    ↓
YOLOv8 (目标检测)
    ↓
ResNext101 (垃圾分类)
    ↓
results/ (检测结果)
```

## 📊 文件大小统计

### 代码文件
- Python源码: ~50KB
- 配置文件: ~5KB

### 模型文件
- YOLOv8模型: ~6MB
- ResNext101模型: ~170MB

### 数据文件
- 训练集: ~11,824张图像
- 验证集: ~2,978张图像
- 总计: ~14,802张图像

### 结果文件
- 检测结果图像: 根据使用情况
- 评估图表: ~1MB

## 🗑️ 可删除文件

以下文件可以安全删除（如需要可重新生成）：

### 临时文件
- `__pycache__/` - Python缓存
- `*.pyc` - 编译的Python文件

### 可重新生成的文件
- `results/detected_*.jpg` - 检测结果（可重新检测）
- `data/processed/dataset_distribution.png` - 数据分布图（可重新生成）
- `models/training_curves.png` - 训练曲线（可重新生成）

### 不需要的文件
- 已删除的临时文档和测试文件

## 📦 备份建议

### 必须备份
- ✅ `models/best_resnext_model.pth` - 训练好的模型
- ✅ `config.py` - 配置文件
- ✅ 所有源代码文件

### 建议备份
- ⚠️ `data/processed/` - 处理后的数据集
- ⚠️ `models/training_history.json` - 训练历史
- ⚠️ `results/` - 评估结果

### 可选备份
- ℹ️ `docs/` - 文档（可从源码重新生成）
- ℹ️ `yolov8n.pt` - YOLOv8模型（可重新下载）

## 🔧 维护建议

### 定期清理
```bash
# 清理Python缓存
find . -type d -name "__pycache__" -exec rm -rf {} +

# 清理临时检测结果
rm results/detected_*.jpg
rm results/demo_*.jpg
```

### 版本控制
建议使用Git管理代码，`.gitignore` 已配置忽略：
- `__pycache__/`
- `*.pyc`
- `data/raw/`
- `models/*.pth`
- `yolov8n.pt`

## 📝 文件命名规范

### 代码文件
- 模块: `snake_case.py`
- 类: `PascalCase`
- 函数: `snake_case()`

### 文档文件
- 英文: `UPPERCASE.md`
- 中文: `中文名称.md`

### 结果文件
- 检测结果: `detected_*.jpg`
- 演示结果: `demo_*.jpg`
- 评估图表: `描述性名称.png`

## 🎯 快速定位

### 需要修改配置？
→ `config.py`

### 需要训练模型？
→ `main.py --step 2`

### 需要检测图像？
→ `demo.py` 或 `main.py --step 4`

### 需要查看文档？
→ `docs/` 目录

### 需要查看结果？
→ `results/` 目录

### 需要修改检测逻辑？
→ `utils/detector.py`

---

**文档版本**: v2.0  
**最后更新**: 2026年1月8日  
**维护状态**: ✅ 活跃维护
