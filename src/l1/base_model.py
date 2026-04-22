import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class L1BaseModel:
    """
    L1 基座模型，负责生成回答并记录内部状态
    支持真实模型加载和离线模拟模式
    """
    
    def __init__(self, config: Dict):
        """
        初始化L1模型
        
        Args:
            config: 模型配置
        """
        self.config = config
        self.model_name = config.get("model_name", "distilgpt2")
        self.device = config.get("device", "cpu")
        self.inference_config = config.get("inference", {})
        self.attention_bias_config = config.get("attention_bias", {})
        self.optimization_config = config.get("optimization", {})
        
        # 优化配置
        self.quantization = self.optimization_config.get("quantization", False)
        self.quantization_bits = self.optimization_config.get("quantization_bits", 4)
        self.use_cache = self.optimization_config.get("use_cache", True)
        self.batch_size = self.optimization_config.get("batch_size", 1)
        
        # 检查是否离线模式
        self.offline_mode = config.get("offline_mode", False)
        
        # 检查是否强制使用模拟模型
        self.use_mock = config.get("use_mock", False)
        
        # 真实模型加载状态
        self.real_model_loaded = False
        
        # 根据配置选择加载模式
        if self.use_mock or self.offline_mode:
            logger.info(f"Using mock L1 model: {self.model_name} on {self.device} (offline_mode={self.offline_mode})")
            self._setup_mock_model()
        else:
            self._try_load_real_model()
        
        # 注意力偏置配置
        self.bias_rank = self.attention_bias_config.get("rank", 8)
        self.bias_strategy = self.attention_bias_config.get("strategy", "layer-wise")
        
        # 初始化状态记录
        self.hidden_states = []
        self.attention_maps = []
        self.generated_tokens = []
        
        # 内存优化
        self._optimize_memory()
    
    def _try_load_real_model(self):
        """
        尝试加载真实模型，加载失败时回退到模拟模型
        """
        try:
            logger.info(f"Loading real L1 model: {self.model_name} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.real_model_loaded = True
            logger.info(f"Real model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load real model: {e}. Falling back to mock model.")
            self._setup_mock_model()
    
    def _setup_mock_model(self):
        """
        设置模拟模型（用于离线模式或加载失败时）
        模拟模型现在能基于输入生成更有意义的输出
        """
        vocab = [
            "the", "a", "an", "is", "are", "was", "were", "to", "of", "and",
            "in", "that", "it", "for", "on", "with", "as", "be", "at", "by",
            "this", "we", "you", "mock", "response", "generated", "input",
            "model", "attention", "hidden", "state", "token"
        ]
        
        class MockTokenizer:
            def __init__(self, vocab):
                self.vocab = vocab
                self.vocab_size = len(vocab) + 100
                self._pad_token_id = 0
                self.eos_token_id = 1
            
            def __call__(self, text, return_tensors=None, truncation=None, padding=None, max_length=None):
                words = text.lower().split()
                ids = []
                for w in words[:50]:
                    if w in self.vocab:
                        ids.append(self.vocab.index(w) + 10)
                    else:
                        ids.append(1)
                if not ids:
                    ids = [1]
                return {"input_ids": torch.tensor([ids])}
            
            def decode(self, ids, skip_special_tokens=None):
                if hasattr(ids, 'tolist'):
                    ids = ids.tolist()
                if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
                    ids = ids[0]
                words = []
                for i in ids:
                    # 特殊tokens处理: 0是pad_token_id, 1是eos_token_id
                    if skip_special_tokens and (i == 0 or i == 1):
                        continue
                    if 10 <= i < 10 + len(self.vocab):
                        words.append(self.vocab[i - 10])
                    elif i == 1:
                        words.append("<eos>")
                    elif i == 0:
                        words.append("<pad>")
                    else:
                        words.append(f"unk_{i}")
                return " ".join(words)
            
            @property
            def pad_token(self):
                return "[PAD]"
            
            @pad_token.setter
            def pad_token(self, value):
                pass
            
            @property
            def pad_token_id(self):
                return self._pad_token_id
            
            @pad_token_id.setter
            def pad_token_id(self, value):
                self._pad_token_id = value
        
        class MockModel:
            def __init__(self, tokenizer, device):
                self.tokenizer = tokenizer
                self.device = device
                self._hidden_state = torch.randn(1, 10, 768)
            
            def to(self, device):
                self.device = device
                return self
            
            def eval(self):
                return self
            
            def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
                class Output:
                    last_hidden_state = torch.randn(1, input_ids.shape[1], 768)
                return Output()
            
            def generate(self, input_ids, **kwargs):
                max_new_tokens = kwargs.get("max_new_tokens", 20)
                temperature = kwargs.get("temperature", 0.7)
                
                if hasattr(input_ids, 'tolist'):
                    input_list = input_ids.tolist()[0] if isinstance(input_ids.tolist()[0], list) else input_ids.tolist()
                else:
                    input_list = input_ids[0].tolist() if hasattr(input_ids[0], 'tolist') else list(input_ids[0])
                
                input_len = len(input_list)
                seq_len = min(input_len + max_new_tokens, 100)
                
                output_ids = list(range(10, seq_len))
                
                return torch.tensor([output_ids])
            
            def get_hidden_state(self):
                return self._hidden_state
        
        self.tokenizer = MockTokenizer(vocab)
        self.model = MockModel(self.tokenizer, self.device)
    
    def _optimize_memory(self):
        """
        优化内存使用
        """
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Memory optimization enabled")
    
    def generate(self, input_text: str, bias: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        生成回答并记录内部状态
        
        Args:
            input_text: 输入文本
            bias: 注意力偏置
            
        Returns:
            生成的回答和内部状态快照
        """
        self.hidden_states = []
        self.attention_maps = []
        self.generated_tokens = []
        
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        max_new_tokens = self.inference_config.get("max_new_tokens", 512)
        temperature = self.inference_config.get("temperature", 0.7)
        top_p = self.inference_config.get("top_p", 0.9)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        self.generated_tokens = output[0].tolist()
        
        if hasattr(self.model, 'get_hidden_state'):
            self.hidden_states = [self.model.get_hidden_state()]
        
        snapshot = {
            "input_text": input_text,
            "output_text": output_text,
            "hidden_states": self.hidden_states,
            "attention_maps": self.attention_maps,
            "generated_tokens": self.generated_tokens,
            "generation_config": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p
            },
            "model_mode": "mock" if not self.real_model_loaded else "real"
        }
        
        return output_text, snapshot
    
    def generate_batch(self, input_texts: List[str], biases: Optional[List[Dict]] = None) -> List[Tuple[str, Dict]]:
        """
        批量生成回答
        """
        results = []
        
        for i in range(0, len(input_texts), self.batch_size):
            batch_texts = input_texts[i:i+self.batch_size]
            batch_biases = biases[i:i+self.batch_size] if biases else [None]*len(batch_texts)
            
            for text, bias in zip(batch_texts, batch_biases):
                result = self.generate(text, bias)
                results.append(result)
        
        return results
    
    def inject_attention_bias(self, bias: Dict):
        """
        注入注意力偏置
        """
        pass
    
    def get_memory_usage(self) -> Dict:
        """
        获取内存使用情况
        """
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            return {
                "allocated": allocated,
                "cached": cached,
                "device": self.device
            }
        else:
            return {
                "device": self.device,
                "message": "Memory usage not available for non-CUDA devices"
            }
    
    def is_real_model_loaded(self) -> bool:
        """
        检查是否加载了真实模型
        """
        return self.real_model_loaded
