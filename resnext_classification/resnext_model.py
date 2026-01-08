"""
ResNext101分类模型
"""
import torch
import torch.nn as nn
import torchvision.models as models

class ResNextClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        初始化ResNext101分类器
        """
        super(ResNextClassifier, self).__init__()
        
        # 加载预训练的ResNext101模型
        self.resnext = models.resnext101_32x8d(pretrained=pretrained)
        
        # 获取最后一层的输入特征数
        num_features = self.resnext.fc.in_features
        
        # 替换最后的全连接层
        self.resnext.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnext(x)
    
    def freeze_backbone(self):
        """
        冻结backbone层，只训练分类头
        """
        for param in self.resnext.parameters():
            param.requires_grad = False
        
        # 解冻最后的全连接层
        for param in self.resnext.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """
        解冻所有层
        """
        for param in self.resnext.parameters():
            param.requires_grad = True
