"""
配置文件 - 垃圾分类系统
"""
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# 原始数据集路径（YOLO格式）
ORIGINAL_DATASET_DIR = r"E:\garbage_classify"
YOLO_TRAIN_DATA_DIR = os.path.join(ORIGINAL_DATASET_DIR, 'train_data')
YOLO_LABEL_FILE = os.path.join(ORIGINAL_DATASET_DIR, 'garbage_classify_rule.json')

# 数据集路径
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')  # 原始数据集
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')  # 处理后的数据
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, 'train')
VAL_DIR = os.path.join(PROCESSED_DATA_DIR, 'val')

# YOLOv8配置
YOLO_MODEL = 'yolov8s.pt'  # YOLOv8 small模型
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45

# ResNext101配置
RESNEXT_PRETRAINED = True
NUM_CLASSES = 40  # 40个细分类别
NUM_MAIN_CLASSES = 4  # 4个主类别
BATCH_SIZE = 32
NUM_EPOCHS = 50  # 保持50个epochs
LEARNING_RATE = 0.001
IMG_SIZE = 224

# 早停配置
EARLY_STOPPING_PATIENCE = 10  # 10个epoch没有提升就停止
EARLY_STOPPING_MIN_DELTA = 0.001  # 最小改善阈值

# 训练配置
TRAIN_SPLIT = 0.8  # 训练集比例
VAL_SPLIT = 0.2    # 验证集比例
RANDOM_SEED = 42

# 设备配置
DEVICE = 'cuda'  # 'cuda' 或 'cpu'

# Python环境
PYTHON_PATH = r"C:\Users\Q\miniconda3\envs\pytorch\python.exe"

# 垃圾分类类别（根据实际情况调整）
# 40个细分类别
CLASS_NAMES = [
    '其他垃圾/一次性快餐盒', '其他垃圾/污损塑料', '其他垃圾/烟蒂', '其他垃圾/牙签',
    '其他垃圾/破碎花盆及碟碗', '其他垃圾/竹筷',
    '厨余垃圾/剩饭剩菜', '厨余垃圾/大骨头', '厨余垃圾/水果果皮', '厨余垃圾/水果果肉',
    '厨余垃圾/茶叶渣', '厨余垃圾/菜叶菜根', '厨余垃圾/蛋壳', '厨余垃圾/鱼骨',
    '可回收物/充电宝', '可回收物/包', '可回收物/化妆品瓶', '可回收物/塑料玩具',
    '可回收物/塑料碗盆', '可回收物/塑料衣架', '可回收物/快递纸袋', '可回收物/插头电线',
    '可回收物/旧衣服', '可回收物/易拉罐', '可回收物/枕头', '可回收物/毛绒玩具',
    '可回收物/洗发水瓶', '可回收物/玻璃杯', '可回收物/皮鞋', '可回收物/砧板',
    '可回收物/纸板箱', '可回收物/调料瓶', '可回收物/酒瓶', '可回收物/金属食品罐',
    '可回收物/锅', '可回收物/食用油桶', '可回收物/饮料瓶',
    '有害垃圾/干电池', '有害垃圾/软膏', '有害垃圾/过期药物'
]

# 4个主类别
MAIN_CLASS_NAMES = ['其他垃圾', '厨余垃圾', '可回收物', '有害垃圾']

# 类别ID到主类别的映射
CLASS_TO_MAIN_CLASS = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,  # 其他垃圾 (0-5)
    6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1,  # 厨余垃圾 (6-13)
    14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2, 20: 2, 21: 2,  # 可回收物 (14-36)
    22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2,
    30: 2, 31: 2, 32: 2, 33: 2, 34: 2, 35: 2, 36: 2,
    37: 3, 38: 3, 39: 3  # 有害垃圾 (37-39)
}
