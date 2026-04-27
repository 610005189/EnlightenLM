import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import json

class HallucinationDataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
    
    def load_data(self):
        """加载数据集"""
        # 由于网络问题，我们使用生成的示例数据
        # 实际应用中，这里应该使用 datasets 库加载真实数据集
        
        # 加载 TruthfulQA 示例数据
        truthful_qa_data = []
        for i in range(100):
            if i % 2 == 0:
                # 真实回答
                question = f"问题{i}: 中国的首都是哪里？"
                answer = "中国的首都是北京。"
                is_hallucination = 0
            else:
                # 幻觉回答
                question = f"问题{i}: 中国的首都是哪里？"
                answer = "中国的首都是上海。"
                is_hallucination = 1
            
            truthful_qa_data.append({
                "question": question,
                "answer": answer,
                "is_hallucination": is_hallucination
            })
        
        # 加载 HaluEval 示例数据
        halu_eval_data = []
        for i in range(100):
            # 生成不同类型的幻觉
            if i % 2 == 0:
                # 真实文本
                text = f"地球的直径约为12742公里。"
                is_hallucination = 0
            else:
                # 幻觉文本
                text = f"地球的直径是10000公里。"
                is_hallucination = 1
            
            halu_eval_data.append({
                "text": text,
                "is_hallucination": is_hallucination
            })
        
        return truthful_qa_data, halu_eval_data
    
    def preprocess(self, dataset):
        """预处理数据，提取特征"""
        features = []
        labels = []
        
        for example in dataset:
            # 提取文本
            if "text" in example:
                text = example["text"]
            else:
                text = example.get("question", "") + " " + example.get("answer", "")
            
            # 计算文本特征
            diversity = self.calculate_diversity(text)
            repetition_rate = self.calculate_repetition_rate(text)
            char_entropy = self.calculate_char_entropy(text)
            length = len(text)
            word_count = len(text.split())
            avg_word_length = word_count / len(text) if len(text) > 0 else 0
            punctuation_ratio = sum(1 for c in text if c in ".,!?;:" ) / len(text) if len(text) > 0 else 0
            uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
            
            # 提取标签
            label = example.get("is_hallucination", 0)
            
            # 构建特征向量
            feature_vector = [
                diversity,
                repetition_rate,
                char_entropy,
                length,
                word_count,
                avg_word_length,
                punctuation_ratio,
                uppercase_ratio
            ]
            
            features.append(feature_vector)
            labels.append(label)
        
        # 标准化特征
        features = np.array(features)
        features = self.scaler.fit_transform(features)
        
        return features, np.array(labels)
    
    def calculate_diversity(self, text):
        """计算文本多样性"""
        tokens = text.split()
        if len(tokens) == 0:
            return 0
        return len(set(tokens)) / len(tokens)
    
    def calculate_repetition_rate(self, text):
        """计算文本重复率"""
        tokens = text.split()
        if len(tokens) < 2:
            return 0
        return 1 - len(set(tokens)) / len(tokens)
    
    def calculate_char_entropy(self, text):
        """计算字符熵"""
        if len(text) == 0:
            return 0
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        entropy = 0
        for count in char_counts.values():
            prob = count / len(text)
            entropy -= prob * np.log2(prob)
        return entropy
    
    def get_scaler(self):
        """获取标准化器"""
        return self.scaler