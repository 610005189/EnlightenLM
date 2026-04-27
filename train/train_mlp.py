import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
print(f"Added project root to Python path: {project_root}")

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from enlighten.memory.hallucination_discriminator import HallucinationDiscriminator, HallucinationDiscriminatorConfig
from enlighten.memory.dataset import HallucinationDataset
import numpy as np

# 创建模型保存目录
os.makedirs("./models", exist_ok=True)

def train():
    print("开始训练 MLP 幻觉判别器...")
    
    # 加载数据
    dataset = HallucinationDataset("./data")
    truthful_qa, halu_eval = dataset.load_data()
    
    # 预处理数据
    truthful_qa_features, truthful_qa_labels = dataset.preprocess(truthful_qa)
    halu_eval_features, halu_eval_labels = dataset.preprocess(halu_eval)
    
    # 合并数据集
    all_features = np.vstack([truthful_qa_features, halu_eval_features])
    all_labels = np.concatenate([truthful_qa_labels, halu_eval_labels])
    
    # 划分训练集和验证集
    split_idx = int(len(all_features) * 0.8)
    train_features, val_features = all_features[:split_idx], all_features[split_idx:]
    train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]
    
    print(f"训练集大小: {len(train_features)}")
    print(f"验证集大小: {len(val_features)}")
    
    # 创建数据加载器
    train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32), 
                                torch.tensor(train_labels, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_features, dtype=torch.float32), 
                              torch.tensor(val_labels, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 初始化模型
    config = HallucinationDiscriminatorConfig()
    model = HallucinationDiscriminator(config)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    num_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model.forward_batch(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model.forward_batch(inputs).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"./models/hallucination_discriminator_{epoch+1}.pt"
            model.save_model(model_path)
            print(f"Best model saved with val loss: {val_loss:.4f} to {model_path}")
    
    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()