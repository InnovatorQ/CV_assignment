"""
数据预处理和数据集划分模块
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import cv2

class DataPreprocessor:
    def __init__(self, raw_data_dir, output_dir, train_split=0.8):
        """
        初始化数据预处理器
        """
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.train_split = train_split
        self.train_dir = os.path.join(output_dir, 'train')
        self.val_dir = os.path.join(output_dir, 'val')
        
    def split_dataset(self, random_seed=42):
        """
        划分训练集和验证集
        """
        # 创建输出目录
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        
        # 获取所有类别（包括子目录）
        categories = []
        for root, dirs, files in os.walk(self.raw_data_dir):
            # 只处理包含图像文件的目录
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                # 获取相对于raw_data_dir的路径作为类别名
                rel_path = os.path.relpath(root, self.raw_data_dir)
                categories.append(rel_path)
        
        print(f"发现 {len(categories)} 个类别")
        
        stats = {'train': {}, 'val': {}}
        
        for category in tqdm(categories, desc="处理类别"):
            category_path = os.path.join(self.raw_data_dir, category)
            
            # 获取该类别的所有图像
            images = [f for f in os.listdir(category_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(images) == 0:
                continue
            
            # 划分训练集和验证集
            train_imgs, val_imgs = train_test_split(
                images, 
                train_size=self.train_split, 
                random_state=random_seed
            )
            
            # 创建类别目录（扁平化，使用完整路径作为类别名）
            # 将 "主类别\子类别" 转换为 "主类别_子类别" 避免多层目录
            flat_category = category.replace('\\', '_').replace('/', '_')
            train_cat_dir = os.path.join(self.train_dir, flat_category)
            val_cat_dir = os.path.join(self.val_dir, flat_category)
            os.makedirs(train_cat_dir, exist_ok=True)
            os.makedirs(val_cat_dir, exist_ok=True)
            
            # 复制训练集图像
            for img in train_imgs:
                src = os.path.join(category_path, img)
                dst = os.path.join(train_cat_dir, img)
                shutil.copy2(src, dst)
            
            # 复制验证集图像
            for img in val_imgs:
                src = os.path.join(category_path, img)
                dst = os.path.join(val_cat_dir, img)
                shutil.copy2(src, dst)
            
            stats['train'][category] = len(train_imgs)
            stats['val'][category] = len(val_imgs)
        
        # 保存统计信息
        stats_path = os.path.join(self.output_dir, 'dataset_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        
        print("\n数据集划分完成!")
        print(f"训练集: {sum(stats['train'].values())} 张图像")
        print(f"验证集: {sum(stats['val'].values())} 张图像")
        
        return stats
    
    def analyze_dataset(self):
        """
        分析数据集统计信息
        """
        import matplotlib.pyplot as plt
        
        stats_path = os.path.join(self.output_dir, 'dataset_stats.json')
        if not os.path.exists(stats_path):
            print("请先运行 split_dataset()")
            return
        
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        # 绘制统计图
        categories = list(stats['train'].keys())
        train_counts = [stats['train'][cat] for cat in categories]
        val_counts = [stats['val'][cat] for cat in categories]
        
        # 创建简化的英文标签（使用索引）
        category_labels = [f'Class {i+1}' for i in range(len(categories))]
        
        fig, ax = plt.subplots(figsize=(16, 6))
        x = range(len(categories))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], train_counts, width, label='Training Set')
        ax.bar([i + width/2 for i in x], val_counts, width, label='Validation Set')
        
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Number of Images', fontsize=12)
        ax.set_title('Dataset Distribution Statistics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(category_labels, rotation=90, ha='right', fontsize=8)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dataset_distribution.png'), dpi=150)
        plt.close()
        
        # 保存类别映射文件
        mapping_path = os.path.join(self.output_dir, 'category_mapping.txt')
        with open(mapping_path, 'w', encoding='utf-8') as f:
            f.write("Category Mapping:\n")
            f.write("=" * 60 + "\n")
            for i, cat in enumerate(categories):
                f.write(f"Class {i+1}: {cat}\n")
        
        print("数据集分析完成，图表已保存")
        print(f"类别映射已保存到: {mapping_path}")
