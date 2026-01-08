# 实时检测使用指南

## 🎯 检测功能

训练好的模型现在可以用于实时检测垃圾分类！

## ✨ 最新更新

**已修复的问题**:
1. ✅ **中文显示问题**: 使用PIL + 中文字体替代OpenCV，完美支持中文显示
2. ✅ **边界框绘制**: 添加了彩色边界框，根据置信度显示不同颜色
   - 绿色: 高置信度 (>80%)
   - 橙色: 中等置信度 (60-80%)
   - 红色: 低置信度 (<60%)
3. ✅ **实时性能优化**: 视频和摄像头检测每3-5帧处理一次，提高流畅度

## 🚀 快速开始

### 方法1: 使用主程序
```bash
python main.py --step 4
```

然后选择检测模式：
1. 检测图像
2. 检测视频
3. 摄像头实时检测

### 方法2: 使用演示脚本
```bash
# 检测单张图像
python demo.py path/to/image.jpg
```

## 📸 检测模式详解

### 1. 图像检测
**适用场景**: 单张图片分类

**使用方法**:
```bash
python main.py --step 4
# 选择 1
# 输入图像路径
```

**输出**:
- 在图像上标注分类结果（支持中文）
- 显示彩色边界框（根据置信度）
- 显示类别和置信度
- 保存到 `results/detected_*.jpg`

**示例**:
```
输入: test_image.jpg
输出: 可回收物_饮料瓶 (置信度: 85.3%)
保存: results/detected_test_image.jpg
```

### 2. 视频检测
**适用场景**: 视频文件中的垃圾分类

**使用方法**:
```bash
python main.py --step 4
# 选择 2
# 输入视频路径
```

**特点**:
- 每5帧检测一次（提高速度）
- 实时显示检测结果
- 保存处理后的视频

**输出**:
- 带标注的视频文件
- 保存到 `results/detected_*.mp4`

### 3. 摄像头实时检测
**适用场景**: 实时垃圾分类

**使用方法**:
```bash
python main.py --step 4
# 选择 3
# 输入摄像头ID（默认0）
```

**特点**:
- 实时检测和显示
- 按 'q' 退出
- 不保存视频（可修改代码保存）

**注意**:
- 需要连接摄像头
- 确保摄像头权限已开启

## 🎨 检测结果示例

### 显示格式
```
┌─────────────────────────────┐
│ 可回收物_饮料瓶             │  ← 中文类别名（正常显示）
│ 置信度: 85.3%               │  ← 置信度百分比
│                             │
│  ┌─────────────────────┐   │
│  │                     │   │  ← 彩色边界框
│  │   [图像内容]        │   │    (绿色/橙色/红色)
│  │                     │   │
│  └─────────────────────┘   │
│                             │
└─────────────────────────────┘
```

### 边界框颜色说明
- **绿色边框**: 高置信度 (>80%) - 分类结果可靠
- **橙色边框**: 中等置信度 (60-80%) - 分类结果基本可信
- **红色边框**: 低置信度 (<60%) - 建议人工确认

### 置信度解读
- **> 90%**: 非常确定 ✅
- **80-90%**: 比较确定 ✅
- **70-80%**: 基本确定 ⚠️
- **60-70%**: 不太确定 ⚠️
- **< 60%**: 建议人工确认 ❌

## 💡 使用技巧

### 提高检测准确率
1. **良好的光照**: 确保图像清晰明亮
2. **合适的角度**: 正面拍摄，避免遮挡
3. **单一目标**: 一次只检测一个物体
4. **清晰的背景**: 避免复杂背景干扰

### 处理低置信度结果
如果置信度 < 70%：
1. 重新拍摄（更好的角度/光照）
2. 查看Top-3预测结果
3. 人工确认分类

## 📊 性能指标

### 检测速度
- **GPU (RTX 4060)**: 50-100ms/图像
- **CPU**: 500-1000ms/图像

### 实时性能
- **图像**: 即时
- **视频**: 实时（30fps）
- **摄像头**: 实时

## 🔧 高级用法

### 批量检测图像
创建脚本 `batch_detect.py`:
```python
from utils.detector import GarbageDetector
import os
import config

# 加载模型
detector = GarbageDetector(
    model_path='models/best_resnext_model.pth',
    class_names=config.CLASS_NAMES,
    device='cuda'
)

# 批量检测
image_dir = 'path/to/images'
for img_file in os.listdir(image_dir):
    if img_file.endswith(('.jpg', '.png')):
        img_path = os.path.join(image_dir, img_file)
        pred_class, confidence = detector.predict_image(img_path)
        print(f"{img_file}: {pred_class} ({confidence:.2%})")
```

### 获取Top-K预测
修改 `utils/detector.py` 中的 `predict_image` 方法：
```python
# 获取Top-3预测
probabilities = torch.softmax(outputs, dim=1)
top3_prob, top3_idx = probabilities.topk(3, dim=1)

for i in range(3):
    class_name = self.class_names[top3_idx[0][i].item()]
    confidence = top3_prob[0][i].item()
    print(f"Top {i+1}: {class_name} ({confidence:.2%})")
```

### 保存检测日志
```python
import csv
from datetime import datetime

# 记录检测结果
with open('detection_log.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        datetime.now(),
        image_path,
        pred_class,
        confidence
    ])
```

## 📁 输出文件

### 检测结果
- `results/detected_*.jpg` - 标注后的图像
- `results/detected_*.mp4` - 标注后的视频

### 检测日志（可选）
- `detection_log.csv` - 检测历史记录
- `detection_stats.json` - 统计信息

## ⚠️ 注意事项

### 模型限制
1. **训练数据**: 只能识别训练过的40个类别
2. **准确率**: 平均77%，部分类别可能较低
3. **环境依赖**: 光照、角度会影响结果

### 使用建议
1. **关键应用**: 建议人工复核
2. **批量处理**: 可以自动化，但需要质量检查
3. **实时应用**: 适合辅助决策，不建议完全自动化

## 🎓 实际应用场景

### 1. 智能垃圾桶
```python
# 实时检测并引导投放
while True:
    frame = camera.read()
    pred_class, confidence = detector.predict_image(frame)
    
    if confidence > 0.8:
        display_message(f"请投放到: {pred_class}")
    else:
        display_message("无法识别，请人工分类")
```

### 2. 垃圾分类教育
```python
# 拍照识别，学习分类
image = take_photo()
pred_class, confidence = detector.predict_image(image)

show_result(pred_class, confidence)
show_explanation(pred_class)  # 显示分类原因
```

### 3. 环境监测
```python
# 定期拍照，统计垃圾类型
for location in monitoring_points:
    image = capture_image(location)
    pred_class, confidence = detector.predict_image(image)
    
    log_detection(location, pred_class, confidence)
    update_statistics()
```

## 🐛 故障排除

### 问题1: 模型加载失败
**解决**:
```bash
# 检查模型文件
ls models/best_resnext_model.pth

# 重新训练
python main.py --step 2
```

### 问题2: 检测速度慢
**解决**:
- 确认使用GPU（应显示cuda）
- 降低图像分辨率
- 减少检测频率（视频）

### 问题3: 准确率低
**解决**:
- 改善拍摄条件（光照、角度）
- 确保目标清晰可见
- 查看置信度，低于70%需人工确认

### 问题4: 摄像头无法打开
**解决**:
- 检查摄像头连接
- 尝试不同的camera_id（0, 1, 2...）
- 检查摄像头权限

## 📈 性能优化

### 提速技巧
1. **使用GPU**: 确保CUDA可用
2. **批处理**: 一次处理多张图像
3. **降低分辨率**: 224x224已足够
4. **模型量化**: 转换为INT8（高级）

### 准确率优化
1. **图像预处理**: 增强对比度、去噪
2. **多角度检测**: 拍摄多张图像
3. **集成预测**: 使用多个模型投票

## ✅ 检查清单

使用前确认：
- [x] 模型已训练完成
- [x] 模型文件存在（best_resnext_model.pth）
- [x] GPU可用（可选，但推荐）
- [x] 测试图像/视频已准备
- [x] 摄像头已连接（如需要）

## 🎉 开始检测

准备好了吗？运行：
```bash
python main.py --step 4
```

或者快速测试：
```bash
python demo.py test_image.jpg
```

---

**祝检测顺利！** 🚀
