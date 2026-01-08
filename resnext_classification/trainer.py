"""
ResNext101模型训练器
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np

class ResNextTrainer:
    def __init__(self, model, train_dir, val_dir, batch_size=32, 
                 learning_rate=0.001, device='cuda'):
        """
        初始化训练器
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # 混合精度训练
        self.use_amp = True  # 启用混合精度
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # 数据增强和预处理
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 加载数据集
        self.train_dataset = datasets.ImageFolder(train_dir, transform=self.train_transform)
        self.val_dataset = datasets.ImageFolder(val_dir, transform=self.val_transform)
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  # Windows优化：设为0避免多进程问题
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,  # Windows优化：设为0避免多进程问题
            pin_memory=True
        )
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"训练集: {len(self.train_dataset)} 张图像")
        print(f"验证集: {len(self.val_dataset)} 张图像")
        print(f"类别: {self.train_dataset.classes}")
    
    def train_epoch(self):
        """
        训练一个epoch（使用混合精度）
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='训练中')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 混合精度训练
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """
        验证模型
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='验证中'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc

    def train(self, num_epochs, save_dir='models'):
        """
        训练模型（带早停机制）
        """
        import config
        
        os.makedirs(save_dir, exist_ok=True)
        best_val_acc = 0.0
        
        # 早停机制
        patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 10)
        min_delta = getattr(config, 'EARLY_STOPPING_MIN_DELTA', 0.001)
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_to_idx': self.train_dataset.class_to_idx
                }, os.path.join(save_dir, 'best_resnext_model.pth'))
                print(f"✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                print(f"早停计数: {patience_counter}/{patience}")
            
            # 早停检查
            if patience_counter >= patience:
                print(f"\n早停触发！验证准确率在 {patience} 个epoch内没有提升")
                print(f"最佳验证准确率: {best_val_acc:.2f}%")
                break
        
        # 保存训练历史
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)
        
        # 绘制训练曲线
        self.plot_training_curves(save_dir)
        
        print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
        print(f"实际训练轮数: {epoch+1}/{num_epochs}")
        
        return self.history
    
    def plot_training_curves(self, save_dir):
        """
        绘制训练曲线
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 损失曲线
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
        plt.close()
        
        print(f"训练曲线已保存到 {save_dir}/training_curves.png")
