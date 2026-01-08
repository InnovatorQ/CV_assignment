# 🚀 开始使用 - 垃圾分类系统

## 📋 前置条件

✅ Python环境: `C:\Users\Q\miniconda3\envs\pytorch\python.exe`  
✅ 数据集路径: `E:\garbage_classify`  
✅ 工作目录: `E:\assignment_CV`

## 🎯 快速开始（3步完成）

### 第1步：安装依赖包 ⏱️ 约5分钟

**方法A - 使用批处理脚本（推荐）:**
1. 双击 `quick_start.bat`
2. 选择 `1` - 安装依赖包

**方法B - 使用命令行:**
```bash
C:\Users\Q\miniconda3\envs\pytorch\python.exe -m pip install -r requirements.txt
```

### 第2步：测试环境 ⏱️ 约1分钟

```bash
C:\Users\Q\miniconda3\envs\pytorch\python.exe test_environment.py
```

确保看到 "✓ 环境配置完成" 的提示。

### 第3步：运行完整流程 ⏱️ 约1-3小时（取决于硬件）

**方法A - 使用批处理脚本（最简单）:**
1. 双击 `quick_start.bat`
2. 选择 `7` - 完整流程

**方法B - 使用命令行:**
```bash
# 转换数据集格式
C:\Users\Q\miniconda3\envs\pytorch\python.exe convert_dataset.py

# 执行训练流程
C:\Users\Q\miniconda3\envs\pytorch\python.exe main.py --step all
```

## 📊 流程说明

### 阶段1: 数据集转换 ⏱️ 约5-10分钟
- 将YOLO格式转换为分类格式
- 按40个类别组织数据
- 生成统计信息

**输出:**
- `data/raw/` - 转换后的数据集
- `data/conversion_stats.json` - 统计信息

### 阶段2: 数据预处理 ⏱️ 约2-5分钟
- 划分训练集（80%）和验证集（20%）
- 生成数据分布图
- 数据集分析

**输出:**
- `data/processed/train/` - 训练集
- `data/processed/val/` - 验证集
- `data/processed/dataset_distribution.png` - 分布图

### 阶段3: 模型训练 ⏱️ 约1-3小时
- 加载预训练ResNext101模型
- 在垃圾分类数据集上训练
- 自动保存最佳模型

**输出:**
- `models/best_resnext_model.pth` - 最佳模型
- `models/training_curves.png` - 训练曲线

### 阶段4: 模型评估 ⏱️ 约2-5分钟
- 在验证集上评估
- 生成详细报告
- 绘制混淆矩阵

**输出:**
- `results/classification_report.txt` - 分类报告
- `results/confusion_matrix.png` - 混淆矩阵
- `results/class_accuracies.png` - 各类别准确率

## 🎮 使用训练好的模型

训练完成后，可以进行实时检测：

### 方法1: 使用主程序
```bash
C:\Users\Q\miniconda3\envs\pytorch\python.exe main.py --step 4
```

然后选择：
1. 检测图像
2. 检测视频
3. 摄像头实时检测

### 方法2: 使用演示脚本
```bash
# 检测单张图像
C:\Users\Q\miniconda3\envs\pytorch\python.exe demo.py path/to/image.jpg
```

## 📁 项目文件说明

### 核心文件
- `config.py` - 配置文件（可修改训练参数）
- `convert_dataset.py` - 数据集转换脚本
- `main.py` - 主程序
- `demo.py` - 演示脚本

### 启动脚本
- `quick_start.bat` - Windows批处理脚本（推荐）
- `test_environment.py` - 环境测试
- `test_conversion.py` - 转换测试

### 文档
- `README.md` - 项目说明
- `START_HERE.md` - 快速开始（本文件）
- `USAGE_GUIDE.md` - 详细使用指南
- `DATA_PREPARATION.md` - 数据准备指南
- `PROJECT_OVERVIEW.md` - 项目技术总览

## ⚙️ 配置调整

如需调整训练参数，编辑 `config.py`:

```python
# 训练参数
BATCH_SIZE = 32        # 批次大小（显存不足可改为16或8）
NUM_EPOCHS = 50        # 训练轮数（可增加到100）
LEARNING_RATE = 0.001  # 学习率
IMG_SIZE = 224         # 图像大小

# 设备配置
DEVICE = 'cuda'        # 使用GPU（如无GPU改为'cpu'）
```

## 🐛 常见问题

### Q: 显存不足 (CUDA out of memory)
**A:** 修改 `config.py` 中的 `BATCH_SIZE`，从32改为16或8

### Q: 没有GPU，可以用CPU训练吗？
**A:** 可以，但会很慢。修改 `config.py` 中的 `DEVICE = 'cpu'`

### Q: 训练中断了怎么办？
**A:** 可以从最佳模型继续训练（需要修改代码加载checkpoint）

### Q: 如何提高准确率？
**A:** 
- 增加训练轮数（NUM_EPOCHS = 100）
- 调整学习率（LEARNING_RATE = 0.0001）
- 增加数据增强
- 检查数据质量

### Q: 转换数据集时出错
**A:** 
- 确认数据集路径: `E:\garbage_classify`
- 确认存在 `train_data` 文件夹
- 确认存在 `garbage_classify_rule.json`

## 📈 预期结果

### 数据集规模
- 总图像数: 约15000-20000张
- 40个细分类别
- 4个主类别

### 训练性能
- 训练时间: 1-3小时（GPU）/ 10-20小时（CPU）
- 预期准确率: 85-95%
- 模型大小: 约170MB

### 推理性能
- 单张图像: 50-100ms (GPU) / 500-1000ms (CPU)
- 视频处理: 实时30fps (GPU)

## 🎯 下一步

完成基础训练后，可以：

1. **优化模型**
   - 调整超参数
   - 尝试不同的数据增强
   - 使用学习率调度策略

2. **部署应用**
   - 开发Web界面
   - 移动端应用
   - API服务

3. **扩展功能**
   - 多目标检测
   - 实时视频流处理
   - 数据库集成

## 📚 更多信息

- 详细使用说明: `USAGE_GUIDE.md`
- 技术细节: `PROJECT_OVERVIEW.md`
- 数据准备: `DATA_PREPARATION.md`

## 💡 提示

- 首次运行会自动下载预训练模型（约170MB）
- 建议使用GPU训练以节省时间
- 训练过程中可以随时按Ctrl+C中断
- 所有结果都会自动保存

---

**准备好了吗？双击 `quick_start.bat` 开始吧！** 🚀
