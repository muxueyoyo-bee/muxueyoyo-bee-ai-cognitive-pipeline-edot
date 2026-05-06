# QClaw 分析管线核心逻辑

QClaw（游戏内容分析系统）的 Multi-Agent 流水线核心代码。

## 架构

```
任务输入 → Locator (D) → Fixer (E→A) → Reviewer (A) → 产出报告
              ↑                            ↑
         KnowledgeBase ←── EDOT 批量诊断 ──┘
```

## 模型路由

- 执行层：DeepSeek V4 Flash（D/E/A 三 Agent）
- 诊断层：DeepSeek V4 Pro（批次完成后触发）
- 自反思缓冲层：已移除（实验证实净效应为零）

## 数据

完整的 22 轮诊断日志和运营报告输出见 `../../logs/` 和 `../../reports/`。

> 本目录存放 QClaw 分析管线的核心代码。完整代码可按需提供或从实验脚本中提取。
