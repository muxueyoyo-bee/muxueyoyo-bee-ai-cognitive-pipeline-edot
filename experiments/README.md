# SWE-bench D-E-A 实验代码

## 实验架构

```
三阶段主线（old36）：
  基线 (V1, 无诊断) → 自反思 (V2, 无诊断) → Mimo批量 (V3, 外部诊断)
                              ↑
跨样本验证（new36）：            │
  基线 (V1, 无诊断)             │ E瓶颈复现
                              │
诊断时机验证（all72）：          │
  实时C1 (V1, V4Pro) → 缓冲C2 (V1, SR→V4Pro)
                              │
分离实验（all72）：             │
  实时 (V2, V4Pro) → 批量 (V2+反馈, V4Pro)
```

## 脚本说明

### `agent_system.py`
核心三 Agent 流水线：Locator (D-解耦) → Fixer (E-评估) → Reviewer (A-聚合)。
包含 KnowledgeBase、Meta-Analyst 自反思、Phase 1/2/3 全流程。

### `separation_experiment.py`
分离实验主脚本。固定 V4Pro + V2 prompts，仅变诊断时机。
- `--realtime-only`：仅跑实时诊断阶段
- `--batch-only`：仅跑批量诊断阶段（需先跑实时）
- `--compare-only`：仅从已有结果构建对比

### `batch_apply_experiment.py`
V4Pro 批量诊断应用实验。将批量审查的系统性发现写入 V2 prompts 后重跑全部 72 实例。

### `turning_point_validation.py`
72 实例两条件验证实验。
- Condition 1: A0 REJECT → V4Pro 实时诊断
- Condition 2: A0 REJECT → Flash 自反思 → 仍 REJECT → V4Pro

### `turning_point_experiment.py`
n=6×2 转折点试点实验。

### `bootstrap_test.py` / `bootstrap_eda_ratio.py`
Bootstrap 统计检验：PASS 率差异检验、E/(D+A) 比率置信区间。

### `full_comparison.py` / `full_separation_analysis.py` / `grand_review.py`
跨阶段/跨条件统计分析报告生成。

### `monte_carlo_simulation.py`
E/(D+A) 比率的蒙特卡洛仿真验证（论文一 §6）。

### `count_data.py`
QClaw 田野数据评论计数脚本（论文四 §3.3）。

## 模型配置

| 组件 | 模型 | API |
|:--|:--|:--|
| 执行模型 (Locator/Fixer/Reviewer) | DeepSeek V4 Flash | `deepseek-v4-flash` |
| 外部诊断 (C1/C2/SEP) | DeepSeek V4 Pro | `deepseek-v4-pro` |
| 外部诊断 (P3) | Mimo 2.5 Pro | Open Code (手动) |

## 数据可用性

完整的 336 次 Agent 运行轨迹、144 条诊断记录、72 例批量 D-E-A 编码因体积原因不包含在仓库中。
需要完整数据集请联系作者，或使用本仓库脚本从头复现（需要 DeepSeek API Key 和 SWE-bench Verified 数据集）。
