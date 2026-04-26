#!/usr/bin/env python3
"""修复hybrid_architecture.py"""

with open('enlighten/hybrid_architecture.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换旧的参数赋值逻辑
old_assignment = '''        # 从配置中读取模型提供者设置
        self.config = config
        self.use_local_model = use_local_model
        self.local_model_name = local_model_name
        self.api_client = api_client
        self.use_bayesian_l3 = use_bayesian_l3
        self.use_l1_adapter = use_l1_adapter
        self.use_skeleton_l2 = use_skeleton_l2

        # 如果配置中包含模型提供者配置，并且没有显式指定参数，则使用配置中的值
        if hasattr(config, 'model_provider'):
            if self.use_local_model is None:
                self.use_local_model = config.model_provider.use_local_model
            if self.local_model_name is None:
                self.local_model_name = config.model_provider.local_model_name

        # 默认值
        if self.use_local_model is None:
            self.use_local_model = False
        if self.local_model_name is None:
            self.local_model_name = "distilgpt2"

        if not self.use_local_model and self.api_client is None:
            # 从配置中读取API提供者设置
            api_provider = "ollama"
            api_model = "qwen2.5:14b"
            if hasattr(config, 'model_provider'):
                api_provider = config.model_provider.api_provider
                api_model = config.model_provider.api_model

            if api_provider == "ollama":
                self.api_client = OllamaAPIClient(OllamaConfig(model=api_model))
            # 可以添加其他API提供者的支持'''

new_assignment = '''        # 从配置中读取模型提供者设置
        self.config = config
        self.api_client = api_client
        self.use_bayesian_l3 = use_bayesian_l3
        self.use_l1_adapter = use_l1_adapter
        self.use_skeleton_l2 = use_skeleton_l2

        # 从config.model_provider读取模型提供者配置
        if hasattr(config, 'model_provider'):
            model_provider = config.model_provider
            self.use_local_model = model_provider.use_local_model
            self.local_model_name = model_provider.local_model_name

            # 如果没有提供api_client，从配置中创建
            if not self.use_local_model and self.api_client is None:
                api_provider = model_provider.api_provider
                api_model = model_provider.api_model

                if api_provider == "ollama":
                    self.api_client = OllamaAPIClient(OllamaConfig(model=api_model))
                # 可以添加其他API提供者的支持
        else:
            # 默认值
            self.use_local_model = False
            self.local_model_name = "distilgpt2"

            if not self.use_local_model and self.api_client is None:
                self.api_client = OllamaAPIClient(OllamaConfig(model="qwen2.5:14b"))'''

content = content.replace(old_assignment, new_assignment)

with open('enlighten/hybrid_architecture.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed hybrid_architecture.py!')