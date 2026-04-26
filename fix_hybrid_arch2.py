#!/usr/bin/env python3
"""修复hybrid_architecture.py第1278-1310行"""

with open('enlighten/hybrid_architecture.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到需要修复的起始行
start_line = None
for i, line in enumerate(lines):
    if '# 从配置中读取模型提供者设置' in line and i > 1270:
        start_line = i
        break

if start_line:
    # 找到需要修复的结束行
    end_line = start_line
    for i in range(start_line, min(start_line + 40, len(lines))):
        if 'self.working_memory = WorkingMemoryManager' in lines[i]:
            end_line = i
            break

    # 新的代码块
    new_code = '''        # 从配置中读取模型提供者设置
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
                self.api_client = OllamaAPIClient(OllamaConfig(model="qwen2.5:14b"))

'''

    # 替换这些行
    lines = lines[:start_line] + [new_code] + lines[end_line:]

    with open('enlighten/hybrid_architecture.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f'Fixed lines {start_line}-{end_line}!')
else:
    print('Could not find the code to fix!')