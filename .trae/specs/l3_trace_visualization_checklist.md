# Checklist

## 文档补充

- [x] docs/L3_signal_mapping.md 包含完整的信号映射表
- [x] 每个映射包含: 论文符号、定义、L2工程实现、计算公式
- [x] 标注了简化近似之处
- [x] 包含内部信号与VAN监控的关联说明

## 数据采集模块

- [x] WorkingMemoryManager.get_l3_trace_signals() 返回完整信号字典
- [x] TraceCollector 类实现三种采集条件
- [x] CSV输出格式正确: timestamp, condition, mu_H, sigma_H2, k_H, p_harm_raw
- [x] HybridEnlightenLM.generate 支持 enable_trace 参数
- [x] HybridEnlightenLM.generate 支持 trace_callback 参数
- [x] 正常条件测试通过
- [x] 噪声注入测试通过
- [x] 偏见注入测试通过

## 可视化脚本

- [x] scatter3d 模式生成3D散点图
- [x] trajectory 模式生成时序轨迹图
- [x] classify 模式生成准确率曲线
- [x] 图表保存为PNG文件
- [x] 图表中文标注正确

## 验证实验

**需要手动运行** - 需要DEEPSEEK_API_KEY环境变量

- [ ] 运行 l3_trace_collector.py 采集数据
- [ ] 正常条件 N=20 实验完成
- [ ] 噪声注入 N=20 实验完成
- [ ] 偏见注入 N=20 实验完成
- [ ] 分类准确率随窗口长度增加
- [ ] L=10 时准确率 > 85% (或记录实际值)

## 运行验证命令

```bash
# 1. 采集数据 (需要 DEEPSEEK_API_KEY)
python tests/l3_trace_collector.py

# 2. 生成3D散点图
python tests/l3_visualization.py --mode scatter3d --data results/l3_trace.csv

# 3. 生成时序轨迹图
python tests/l3_visualization.py --mode trajectory --data results/l3_trace.csv

# 4. 评估分类准确率
python tests/l3_visualization.py --mode classify --data results/l3_trace.csv
```
