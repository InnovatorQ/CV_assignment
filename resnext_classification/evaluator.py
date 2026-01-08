"""
模型评估模块
"""
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, model, val_loader, class_names, device='cuda'):
        """
        初始化评估器
        """
        self.model = model.to(device)
        self.val_loader = val_loader
        self.class_names = class_names
        self.device = device
    
    def evaluate(self):
        """
        评估模型性能
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='评估中'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 计算准确率
        accuracy = 100. * np.sum(all_preds == all_labels) / len(all_labels)
        
        # 生成分类报告
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=self.class_names,
            digits=4
        )
        
        # 生成混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        
        return accuracy, report, cm, all_preds, all_labels
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        绘制混淆矩阵
        """
        # 使用简化的类别标签
        class_labels = [f'C{i+1}' for i in range(len(self.class_names))]
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"混淆矩阵已保存到 {save_path}")
            
            # 保存类别映射
            mapping_path = save_path.replace('confusion_matrix.png', 'class_mapping.txt')
            with open(mapping_path, 'w', encoding='utf-8') as f:
                f.write("Class Mapping for Confusion Matrix:\n")
                f.write("=" * 60 + "\n")
                for i, name in enumerate(self.class_names):
                    f.write(f"C{i+1}: {name}\n")
            print(f"类别映射已保存到 {mapping_path}")
        
        plt.close()
    
    def analyze_errors(self, all_preds, all_labels, save_path=None):
        """
        分析错误分类
        """
        # 计算每个类别的准确率
        class_accuracies = {}
        for i, class_name in enumerate(self.class_names):
            mask = all_labels == i
            if mask.sum() > 0:
                acc = 100. * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
                class_accuracies[class_name] = acc
        
        # 使用简化标签绘制
        class_labels = [f'Class {i+1}' for i in range(len(self.class_names))]
        accuracies = list(class_accuracies.values())
        
        plt.figure(figsize=(16, 6))
        bars = plt.bar(range(len(class_labels)), accuracies, color='steelblue', alpha=0.8)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Classification Accuracy by Category', fontsize=14, fontweight='bold')
        plt.xticks(range(len(class_labels)), class_labels, rotation=90, ha='right', fontsize=8)
        plt.ylim([0, 100])
        plt.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, v) in enumerate(zip(bars, accuracies)):
            plt.text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.1f}%', 
                    ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"类别准确率图已保存到 {save_path}")
            
            # 保存详细的准确率数据
            acc_data_path = save_path.replace('class_accuracies.png', 'class_accuracies.txt')
            with open(acc_data_path, 'w', encoding='utf-8') as f:
                f.write("Classification Accuracy by Category:\n")
                f.write("=" * 60 + "\n")
                for i, (name, acc) in enumerate(class_accuracies.items()):
                    f.write(f"Class {i+1} ({name}): {acc:.2f}%\n")
            print(f"详细准确率数据已保存到 {acc_data_path}")
        
        plt.close()
        
        return class_accuracies
    
    def full_evaluation(self, save_dir='results'):
        """
        完整评估并保存结果
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("开始模型评估...")
        accuracy, report, cm, all_preds, all_labels = self.evaluate()
        
        print(f"\n总体准确率: {accuracy:.2f}%")
        print("\n分类报告:")
        print(report)
        
        # 保存分类报告
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write(f"总体准确率: {accuracy:.2f}%\n\n")
            f.write("分类报告:\n")
            f.write(report)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(cm, os.path.join(save_dir, 'confusion_matrix.png'))
        
        # 分析错误
        class_acc = self.analyze_errors(
            all_preds, 
            all_labels, 
            os.path.join(save_dir, 'class_accuracies.png')
        )
        
        print("\n各类别准确率:")
        for class_name, acc in class_acc.items():
            print(f"  {class_name}: {acc:.2f}%")
        
        print(f"\n评估结果已保存到 {save_dir}")
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'class_accuracies': class_acc
        }
