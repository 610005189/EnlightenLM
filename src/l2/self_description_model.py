import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class L2SelfDescriptionModel:
    """
    L2 自描述小模型，负责基于L1的快照生成元描述
    """
    
    def __init__(self, config: Dict):
        """
        初始化L2模型
        
        Args:
            config: 模型配置
        """
        self.config = config
        self.model_name = config.get("model_name", "distilbert-base-uncased")
        self.device = config.get("device", "cuda")
        self.inference_config = config.get("inference", {})
        
        # 加载模型和分词器
        logger.info(f"Loading L2 model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        # 元指令模板
        self.meta_instruction = """
        以下是另一个模型在**过去**生成回答时留下的内部记录。请基于这些记录，描述该模型**当时**的推理路径、依赖的关键信息及可能的局限。注意：你正在进行的描述行为本身**不属于**记录内容，请勿描述你此刻的思考过程，勿使用"我正在分析……"等表述。
        
        [输入问题]
        {input_text}
        
        [生成回答]
        {output_text}
        
        [注意力模式摘要]
        {attention_summary}
        
        [生成统计]
        - 生成步数：{generation_steps}
        - 温度：{generation_temperature}
        """
    
    def generate_meta_description(self, snapshot: Dict) -> str:
        """
        基于快照生成元描述
        
        Args:
            snapshot: L1生成的快照
            
        Returns:
            元描述文本
        """
        # 构建注意力摘要
        attention_summary = self._build_attention_summary(snapshot.get("attention_summaries", []))
        
        # 构建输入提示
        prompt = self.meta_instruction.format(
            input_text=snapshot.get("input_text", ""),
            output_text=snapshot.get("output_text", ""),
            attention_summary=attention_summary,
            generation_steps=snapshot.get("generation_steps", 0),
            generation_temperature=snapshot.get("generation_temperature", 0.7)
        )
        
        # 分词
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        # 生成配置
        max_new_tokens = self.inference_config.get("max_new_tokens", 256)
        temperature = self.inference_config.get("temperature", 0.6)
        
        # 生成元描述
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
        
        # 解码生成的文本
        meta_description = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        return meta_description
    
    def _build_attention_summary(self, attention_summaries: list) -> str:
        """
        构建注意力摘要
        
        Args:
            attention_summaries: 注意力摘要列表
            
        Returns:
            注意力摘要文本
        """
        if not attention_summaries:
            return "- 无注意力数据"
        
        summary_parts = []
        
        # 计算平均注意力熵
        avg_entropies = [summary.get("avg_entropy", 0) for summary in attention_summaries]
        if avg_entropies:
            avg_entropy = sum(avg_entropies) / len(avg_entropies)
            summary_parts.append(f"- 平均注意力熵：{avg_entropy:.2f}")
        
        # 提取最后一个层的top tokens
        if attention_summaries:
            last_layer = attention_summaries[-1]
            top_tokens = last_layer.get("top_tokens", [])
            if top_tokens:
                # 取最后几个位置的top tokens
                recent_tokens = top_tokens[-3:] if len(top_tokens) > 3 else top_tokens
                for token_info in recent_tokens:
                    position = token_info.get("position", 0)
                    tokens = token_info.get("tokens", [])
                    weights = token_info.get("weights", [])
                    if tokens and weights:
                        # 只取前2个token
                        top_2 = list(zip(tokens[:2], weights[:2]))
                        token_str = ", ".join([f"{t}({w:.2f})" for t, w in top_2])
                        summary_parts.append(f"- 生成第 {position} 步时最关注 token：{token_str}")
        
        # 跨句依赖强度
        summary_parts.append("- 跨句依赖强度：中等")
        
        return "\n".join(summary_parts)
