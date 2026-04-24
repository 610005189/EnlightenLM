# Tasks

## 任务1: 补充第一阶段信号映射文档

- [x] Task 1.1: 创建 docs/L3_signal_mapping.md
  - 建立论文符号到L2输出的映射表
  - 说明每个映射的工程实现方式
  - 标注简化近似之处

- [x] Task 1.2: 验证映射完整性
  - 确认论文4个内部信号都有对应工程量
  - 补充缺失的近似计算公式

## 任务2: 实现Trace数据采集模块

- [x] Task 2.1: 扩展 WorkingMemoryManager
  - 新增 get_l3_trace_signals() 返回完整信号字典
  - 包含: mu_H, sigma_H2, k_H, p_harm_raw

- [x] Task 2.2: 创建 tests/l3_trace_collector.py
  - TraceCollector 类封装数据采集逻辑
  - 支持三种条件: normal, noise_injection, bias_injection
  - 输出CSV格式: timestamp, condition, mu_H, sigma_H2, k_H, p_harm_raw

- [x] Task 2.3: 修改 HybridEnlightenLM.generate
  - 添加 enable_trace 参数
  - 添加 trace_callback 参数
  - 推理后调用回调记录信号

- [x] Task 2.4: 创建测试用例
  - 测试正常条件采集
  - 测试噪声注入采集
  - 测试偏见注入采集
  - 12个测试全部通过

## 任务3: 实现可视化脚本

- [x] Task 3.1: 创建 tests/l3_visualization.py
  - scatter3d 模式: 3D散点图
  - trajectory 模式: 时序轨迹
  - classify 模式: 分类准确率曲线

- [x] Task 3.2: 实现3D散点图功能
  - 使用matplotlib 3D projection
  - 按condition分色
  - 保存为 PNG 文件

- [x] Task 3.3: 实现时序轨迹功能
  - 多条件曲线叠加
  - 标注关键转折点

- [x] Task 3.4: 实现分类评估功能
  - 使用滑动窗口 (L=1,3,5,10,15,20)
  - 计算3类分类准确率
  - 绘制准确率vs窗口长度曲线

## 任务4: 运行验证实验

**需要手动运行** - 需要DEEPSEEK_API_KEY环境变量

- [ ] Task 4.1: 运行正常条件实验 (N=20)
  - 运行: python tests/l3_trace_collector.py
  - 采集数据并保存到 results/

- [ ] Task 4.2: 运行噪声注入实验 (N=20)
  - 采集数据并保存

- [ ] Task 4.3: 运行偏见注入实验 (N=20)
  - 采集数据并保存

- [ ] Task 4.4: 验证可区分性
  - 运行: python tests/l3_visualization.py --mode classify --data results/l3_trace.csv
  - 验证成功标准: L=10时准确率>85%

## Task Dependencies
- Task 1.1 和 Task 1.2 可并行执行 ✅
- Task 2 依赖 Task 1 完成 ✅
- Task 3 依赖 Task 2 完成 ✅
- Task 4 依赖 Task 2 和 Task 3 完成 ✅

## 完成状态
- [x] 任务1: 100% 完成
- [x] 任务2: 100% 完成
- [x] 任务3: 100% 完成
- [ ] 任务4: 待用户手动运行实验
