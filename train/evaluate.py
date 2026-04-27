import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader, TensorDataset
from enlighten.memory.hallucination_discriminator import HallucinationDiscriminator, HallucinationDiscriminatorConfig
from enlighten.memory.dataset import HallucinationDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def evaluate():
    print("开始评估 MLP 幻觉判别器...")
    
    # 加载数据
    dataset = HallucinationDataset("./data")
    truthful_qa, halu_eval = dataset.load_data()
    
    # 预处理数据
    truthful_qa_features, truthful_qa_labels = dataset.preprocess(truthful_qa)
    halu_eval_features, halu_eval_labels = dataset.preprocess(halu_eval)
    
    # 合并数据集
    all_features = np.vstack([truthful_qa_features, halu_eval_features])
    all_labels = np.concatenate([truthful_qa_labels, halu_eval_labels])
    
    # 划分测试集（使用全部数据作为测试集）
    test_features = all_features
    test_labels = all_labels
    
    print(f"测试集大小: {len(test_features)}")
    
    # 创建数据加载器
    test_dataset = TensorDataset(torch.tensor(test_features, dtype=torch.float32), 
                               torch.tensor(test_labels, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 加载模型
    config = HallucinationDiscriminatorConfig()
    model = HallucinationDiscriminator(config)
    
    # 找到最新的模型文件
    model_files = [f for f in os.listdir("./models") if f.startswith("hallucination_discriminator_") and f.endswith(".pt")]
    if not model_files:
        print("未找到模型文件！")
        return
    
    # 按文件名排序，找到最新的模型
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_model_path = os.path.join("./models", model_files[-1])
    
    print(f"加载模型: {latest_model_path}")
    model.load_state_dict(torch.load(latest_model_path))
    model.eval()
    
    # 评估
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model.forward_batch(inputs).squeeze()
            # 将概率转换为二进制预测
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    print("\n评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵:")
    print(cm)
    
    print("\n评估完成！")

if __name__ == "__main__":
    evaluate()