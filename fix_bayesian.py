#!/usr/bin/env python3
"""修复api_server.py，禁用BayesianL3"""

with open('enlighten/api_server.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 禁用BayesianL3
content = content.replace('use_bayesian_l3=True', 'use_bayesian_l3=False')

with open('enlighten/api_server.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Disabled BayesianL3!')