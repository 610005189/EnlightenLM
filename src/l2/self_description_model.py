import torch
import logging
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class L2SelfDescriptionModel:
    """
    L2 自描述小模型，负责基于L1的快照生成元描述
    
    重要更新：
    - 现在使用生成式模型而不是分类模型来生成元描述
    - 正确处理L1快照中的注意力摘要信息
    - 支持真实模型和模拟模型两种模式
    """
    
    def __init__(self, config: Dict):
        """
        初始化L2模型
        
        Args:
            config: 模型配置
        """
        self.config = config
        self.model_name = config.get("model_name", "distilgpt2")
        self.device = config.get("device", "cpu")
        self.inference_config = config.get("inference", {})
        
        # 检查是否使用模拟模式
        self.offline_mode = config.get("offline_mode", False)
        self.use_mock = config.get("use_mock", False)
        
        if self.use_mock or self.offline_mode:
            logger.info(f"Using mock L2 model: {self.model_name}")
            self._setup_mock_model()
        else:
            self._try_load_real_model()
    
    def _try_load_real_model(self):
        """
        尝试加载真实的生成式模型
        """
        try:
            logger.info(f"Loading real L2 model: {self.model_name} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # 使用因果语言模型（生成式模型）而不是分类模型
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Real L2 model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load real L2 model: {type(e).__name__}: {str(e)}. Falling back to mock model.")
            self._setup_mock_model()
    
    def _setup_mock_model(self):
        """
        设置模拟模型
        """
        class MockTokenizer:
            def __init__(self):
                self.vocab_size = 50000
                self.pad_token_id = 0
                self.eos_token_id = 1
            
            def __call__(self, text, return_tensors=None, truncation=None, padding=None, max_length=None):
                return {
                    "input_ids": torch.tensor([list(range(2, 22))]),
                    "attention_mask": torch.tensor([[1]*20])
                }
            
            def decode(self, ids, skip_special_tokens=None):
                return "This is a generated meta description of the model's reasoning process."
        
        class MockModel:
            def __init__(self, device):
                self.device = device
            
            def to(self, device):
                return self
            
            def eval(self):
                return self
            
            def generate(self, input_ids, **kwargs):
                return torch.tensor([list(range(2, 32))])
        
        self.tokenizer = MockTokenizer()
        self.model = MockModel(self.device)
    
    def _build_meta_prompt(self, snapshot: Dict) -> str:
        """
        从L1快照构建元描述的提示词
        
        Args:
            snapshot: L1生成的快照
            
        Returns:
            完整的提示词
        """
        # 元指令模板
        meta_instruction = """Based on the following internal state snapshot of the model, 
please generate a concise, clear meta description of the reasoning process,
the key information relied upon, and the potential limitations observed.

[Input Question]
{input_text}

[Generated Response]
{output_text}

[Attention Pattern Summary]
{attention_summary}

[Generation Statistics]
- Generation steps: {generation_steps}
- Temperature: {temperature}

[Meta Description]:
"""
        
        # 处理注意力摘要
        attention_summary = self._format_attention_summaries(snapshot.get("attention_summaries", []))
        
        # 构建提示词
        prompt = meta_instruction.format(
            input_text=snapshot.get("input_text", ""),
            output_text=snapshot.get("output_text", ""),
            attention_summary=attention_summary,
            generation_steps=len(snapshot.get("generated_tokens", [])),
            temperature=snapshot.get("generation_config", {}).get("temperature", 0.7)
        )
        
        return prompt
    
    def _format_attention_summaries(self, attention_summaries: list) -> str:
        """
        格式化注意力摘要为字符串
        
        Args:
            attention_summaries: 从L1快照中获取的注意力摘要
            
        Returns:
            格式化的摘要字符串
        """
        if not attention_summaries:
            return "No attention summary available."
        
        summary_parts = []
        
        for summary in attention_summaries:
            layer_num = summary.get("layer", 0)
            avg_entropy = summary.get("avg_entropy", 0.0)
            top_tokens = summary.get("top_tokens", [])
            weights = summary.get("weights", [])
            
            summary_parts.append(f"Layer {layer_num}:")
            summary_parts.append(f"  Avg Entropy: {avg_entropy:.4f}")
            
            if top_tokens and weights:
                tokens_str = ", ".join([f"tok_{t}(w={w:.4f})" for t, w in zip(top_tokens, weights)])
                summary_parts.append(f"  Top Tokens: {tokens_str}")
        
        return "\n".join(summary_parts)
    
    def generate_meta_description(self, snapshot: Dict) -> str:
        """
        基于快照生成元描述
        
        这是修复后的实现：使用生成式模型而不是分类模型
        
        Args:
            snapshot: L1生成的快照
            
        Returns:
            元描述文本
        """
        try:
            # 构建提示词
            prompt = self._build_meta_prompt(snapshot)
            
            # 配置生成参数
            max_new_tokens = self.inference_config.get("max_new_tokens", 150)
            temperature = self.inference_config.get("temperature", 0.7)
            top_p = self.inference_config.get("top_p", 0.9)
            
            # 如果是真实模型，执行生成
            if self.model_name != "distilgpt2" and hasattr(self.tokenizer, "__call__"):
                # 真实模型的情况
                inputs = self.tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # 解码输出（仅解码新生成的部分）
                generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                
            else:
                # 模拟模型或简单情况，使用启发式生成
                generated_text = self._heuristic_meta_description(snapshot)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating meta description: {e}")
            return f"Error generating meta description: {str(e)}"
    
    def _heuristic_meta_description(self, snapshot: Dict) -> str:
        """
        当无法使用真实模型时的启发式元描述生成
        
        Args:
            snapshot: L1快照
            
        Returns:
            启发式生成的元描述
        """
        input_text = snapshot.get("input_text", "")
        output_text = snapshot.get("output_text", "")
        attention_summaries = snapshot.get("attention_summaries", [])
        generation_config = snapshot.get("generation_config", {})
        
        # 构建基础元描述
        meta_parts = []
        
        # 1. 输入处理总结
        if input_text:
            input_len = len(input_text.split())
            meta_parts.append(f"This model processed an input of approximately {input_len} tokens.")
        
        # 2. 注意力模式总结
        if attention_summaries:
            avg_entropy = sum(s.get("avg_entropy", 0) for s in attention_summaries) / len(attention_summaries)
            
            if avg_entropy < 0.5:
                meta_parts.append("The attention pattern appears relatively focused.")
            elif avg_entropy < 1.0:
                meta_parts.append("The attention pattern is moderately distributed.")
            else:
                meta_parts.append("The attention pattern is quite spread out across different parts of the input.")
            
            total_tokens = sum(len(s.get("top_tokens", [])) for s in attention_summaries)
            if total_tokens > 0:
                meta_parts.append(f"Attention was tracked across {len(attention_summaries)} layers.")
        
        # 3. 生成参数总结
        if generation_config:
            temp = generation_config.get("temperature", 0.7)
            meta_parts.append(f"Generation was performed with temperature {temp:.2f}.")
        
        # 4. 输出总结
        if output_text:
            output_len = len(output_text.split())
            meta_parts.append(f"The generated response contains approximately {output_len} tokens.")
        
        # 组合所有部分
        if meta_parts:
            meta_description = " ".join(meta_parts)
            meta_description = meta_description[0].upper() + meta_description[1:] + "."
            return meta_description
        else:
            return "The model's reasoning process was captured for this inference."
