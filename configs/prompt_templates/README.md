# D-E-A / EDOT 标准 Prompt 模板库

按"解耦 / 评估 / 聚合 / 批量诊断 / 记忆注入"五类组织，含占位符 `{PLACEHOLDER}` 与调用说明。所有模板均经 SWE-bench + QClaw 双场景验证。

## 模型映射

| 模板 | 推荐模型 | Temperature |
|:--|:--|:--|
| Locator (D) | DeepSeek V4 Flash | 0.3 |
| Fixer (E→A) | DeepSeek V4 Flash | 0.3 |
| Reviewer (A) | DeepSeek V4 Flash | 0.3 |
| Batch Diagnosis | DeepSeek V4 Pro | 0.5 |
| Memory Injection | — (系统注入) | — |

## 占位符说明

- `{TASK_INPUT}` — 原始任务描述
- `{KB_CONTEXT}` — 长期记忆注入内容（由 memory_injection.yaml 控制）
- `{PREVIOUS_TRAJECTORIES}` — 本批次已完成实例的轨迹摘要
- `{DIAGNOSIS_FOCUS}` — 本轮诊断的焦点领域
