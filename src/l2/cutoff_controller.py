import torch
import numpy as np
from collections import deque
from typing import Dict, List, Optional
import re
import logging

logger = logging.getLogger(__name__)


class CutoffController:
    """
    截断控制器，防止L2模型陷入递归和语义循环
    """
    
    def __init__(self, config: Dict):
        """
        初始化截断控制器
        
        Args:
            config: 控制器配置
        """
        self.config = config
        self.cross_depth_limit = config.get("cross_depth_limit", 1)
        self.semantic_similarity_threshold = config.get("semantic_similarity_threshold", 0.85)
        self.self_reference_weight = config.get("self_reference_weight", 1.0)
        self.self_reference_weight_min = config.get("self_reference_weight_min", 0.1)
        self.phrase_repetition_limit = config.get("phrase_repetition_limit", 3)
        
        # 状态变量
        self.delta_cross = 0
        self.sentence_embeddings = []
        self.recent_tokens = deque(maxlen=50)
        
        # 自反词汇列表
        self.self_reference_words = [
            "我注意到", "我正在", "我分析", "我发现", "上述描述", "刚才的分析", "这个元认知",
            "I notice", "I am analyzing", "the above description"
        ]
        
        # 编译正则表达式
        self.self_reference_patterns = [
            re.compile(r"我\w*正在"),
            re.compile(r"I\s+am\s+\w+ing")
        ]
        
        # 加载句子嵌入模型
        self.embedding_model = None
        self.embedding_tokenizer = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """
        加载句子嵌入模型
        """
        # 模拟嵌入模型，避免网络依赖
        logger.info("Using mock embedding model instead of downloading from Hugging Face")
        
        # 模拟sentence-transformers模型
        class MockEmbeddingModel:
            def encode(self, text):
                # 生成随机嵌入向量
                return np.random.rand(384)
        
        self.embedding_model = MockEmbeddingModel()
        self.embedding_tokenizer = None
    
    def calculate_sentence_embedding(self, sentence: str) -> np.ndarray:
        """
        计算句子嵌入
        
        Args:
            sentence: 句子
            
        Returns:
            句子嵌入向量
        """
        if hasattr(self.embedding_model, 'encode'):
            # 使用sentence-transformers
            return self.embedding_model.encode(sentence)
        else:
            # 使用简单的平均词嵌入
            inputs = self.embedding_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            # 取最后一层隐藏状态的平均值
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return embedding
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        计算两个嵌入向量的余弦相似度
        
        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
            
        Returns:
            余弦相似度
        """
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-10)
    
    def check_cross_depth(self) -> bool:
        """
        检查跨轮次递归深度
        
        Returns:
            是否超过深度限制
        """
        return self.delta_cross >= self.cross_depth_limit
    
    def check_semantic_cycle(self, sentence: str) -> bool:
        """
        检查语义循环
        
        Args:
            sentence: 当前句子
            
        Returns:
            是否检测到语义循环
        """
        if not sentence:
            return False
        
        # 计算当前句子的嵌入
        current_embedding = self.calculate_sentence_embedding(sentence)
        
        # 检查与历史句子的相似度
        cycle_detected = False
        for embedding in self.sentence_embeddings[-2:]:  # 只检查最近两句
            similarity = self.calculate_similarity(current_embedding, embedding)
            if similarity > self.semantic_similarity_threshold:
                cycle_detected = True
                break
        
        # 添加当前句子嵌入到历史
        self.sentence_embeddings.append(current_embedding)
        if len(self.sentence_embeddings) > 10:  # 只保留最近10个句子
            self.sentence_embeddings.pop(0)
        
        return cycle_detected
    
    def check_self_reference(self, text: str) -> bool:
        """
        检查自我指涉词汇
        
        Args:
            text: 文本
            
        Returns:
            是否检测到自我指涉
        """
        # 检查精确匹配
        for word in self.self_reference_words:
            if word in text:
                return True
        
        # 检查正则模式
        for pattern in self.self_reference_patterns:
            if pattern.search(text):
                return True
        
        return False
    
    def check_phrase_repetition(self, text: str) -> bool:
        """
        检查短语重复
        
        Args:
            text: 文本
            
        Returns:
            是否检测到短语重复
        """
        # 简单的短语重复检测
        words = text.split()
        if len(words) < 3:
            return False
        
        # 检查3-gram重复
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            count = text.count(phrase)
            if count >= self.phrase_repetition_limit:
                return True
        
        return False
    
    def check_attention_entropy(self, avg_entropy: float) -> bool:
        """
        检查注意力熵过低
        
        Args:
            avg_entropy: 平均注意力熵
            
        Returns:
            是否注意力熵过低
        """
        return avg_entropy < 0.5
    
    def update_self_reference_weight(self):
        """
        更新自我参照权重
        """
        self.self_reference_weight *= 0.5
        if self.self_reference_weight < self.self_reference_weight_min:
            self.self_reference_weight = self.self_reference_weight_min
    
    def reset(self):
        """
        重置控制器状态
        """
        self.delta_cross = 0
        self.sentence_embeddings = []
        self.recent_tokens.clear()
        self.self_reference_weight = self.config.get("self_reference_weight", 1.0)
    
    def should_cutoff(self, text: str, avg_entropy: Optional[float] = None) -> Dict:
        """
        检查是否应该截断
        
        Args:
            text: 生成的文本
            avg_entropy: 平均注意力熵
            
        Returns:
            截断信息，包含是否截断和原因
        """
        # 检查跨轮次递归
        if self.check_cross_depth():
            return {"cutoff": True, "reason": "cross_depth_exceeded"}
        
        # 检查语义循环
        sentences = re.split(r'[。！？.!?]', text)
        for sentence in sentences:
            if sentence.strip() and self.check_semantic_cycle(sentence.strip()):
                self.update_self_reference_weight()
                if self.self_reference_weight < self.self_reference_weight_min:
                    return {"cutoff": True, "reason": "semantic_cycle_detected"}
        
        # 检查自我指涉
        if self.check_self_reference(text):
            return {"cutoff": True, "reason": "self_reference_detected"}
        
        # 检查注意力熵
        if avg_entropy is not None and self.check_attention_entropy(avg_entropy):
            return {"cutoff": True, "reason": "low_attention_entropy"}
        
        # 检查短语重复
        if self.check_phrase_repetition(text):
            return {"cutoff": True, "reason": "phrase_repetition_detected"}
        
        return {"cutoff": False, "reason": "no_cutoff"}
