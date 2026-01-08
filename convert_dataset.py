"""
将YOLO格式数据集转换为分类格式
"""
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import config

def load_class_mapping():
    """加载类别映射"""
    with open(config.YOLO_LABEL_FILE, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return mapping

def convert_yolo_to_classification():
    """
    将YOLO格式转换为分类格式
    """
    print("=" * 60)
    print("数据集格式转换")
    print("=" * 60)
    
    # 加载类别映射
    class_mapping = load_class_mapping()
    print(f"\n发现 {len(class_mapping)} 个类别")
    
    # 创建输出目录
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    
    # 为每个类别创建目录
    for class_id, class_name in class_mapping.items():
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # 统计信息
    stats = {class_name: 0 for class_name in class_mapping.values()}
    
    # 获取所有标注文件
    label_files = list(Path(config.YOLO_TRAIN_DATA_DIR).glob('*.txt'))
    
    print(f"\n开始转换 {len(label_files)} 个标注文件...")
    
    for label_file in tqdm(label_files, desc="转换中"):
        # 读取标注
        with open(label_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            continue
        
        # 解析标注格式: "img_1.jpg, 0"
        parts = content.split(',')
        if len(parts) != 2:
            continue
        
        img_name = parts[0].strip()
        class_id = parts[1].strip()
        
        # 获取类别名称
        class_name = class_mapping.get(class_id)
        if not class_name:
            print(f"警告: 未知类别ID {class_id}")
            continue
        
        # 源图像路径
        src_img = os.path.join(config.YOLO_TRAIN_DATA_DIR, img_name)
        
        if not os.path.exists(src_img):
            continue
        
        # 目标路径
        dst_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        dst_img = os.path.join(dst_dir, img_name)
        
        # 复制图像
        shutil.copy2(src_img, dst_img)
        stats[class_name] += 1
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("转换完成! 数据集统计:")
    print("=" * 60)
    
    # 按主类别分组统计
    main_class_stats = {
        '其他垃圾': 0,
        '厨余垃圾': 0,
        '可回收物': 0,
        '有害垃圾': 0
    }
    
    for class_name, count in sorted(stats.items()):
        print(f"  {class_name:30s}: {count:5d} 张")
        
        # 累加到主类别
        for main_class in main_class_stats.keys():
            if class_name.startswith(main_class):
                main_class_stats[main_class] += count
                break
    
    print("\n主类别统计:")
    for main_class, count in main_class_stats.items():
        print(f"  {main_class:20s}: {count:5d} 张")
    
    total = sum(stats.values())
    print(f"\n总计: {total} 张图像")
    print("=" * 60)
    
    # 保存统计信息
    stats_file = os.path.join(config.DATA_DIR, 'conversion_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'detail_stats': stats,
            'main_class_stats': main_class_stats,
            'total': total
        }, f, indent=4, ensure_ascii=False)
    
    print(f"\n统计信息已保存到: {stats_file}")
    
    return stats

def main():
    """主函数"""
    # 检查原始数据集
    if not os.path.exists(config.YOLO_TRAIN_DATA_DIR):
        print(f"错误: 找不到数据集目录 {config.YOLO_TRAIN_DATA_DIR}")
        return
    
    if not os.path.exists(config.YOLO_LABEL_FILE):
        print(f"错误: 找不到类别映射文件 {config.YOLO_LABEL_FILE}")
        return
    
    # 转换数据集
    convert_yolo_to_classification()
    
    print("\n下一步:")
    print("  运行: python main.py --step 1")
    print("  或者: python main.py --step all")

if __name__ == '__main__':
    main()
