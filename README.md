# AI Cognitive Pipeline: D-E-A × EDOT

> **你的 Multi-Agent 系统跑着跑着就不进步了？问题不在模型，在你的注意力配置结构。**

一个面向 AI 运营/开发者的开源工具包：识别、诊断、突破 Multi-Agent 系统的"自我迭代天花板"。

---

## 1. 解决什么问题

Multi-Agent 系统部署后会撞上一个天花板：系统**能**自己发现"拆解不够细"（把大任务拆成小任务），但**不能**自己发现"评估标准有问题"（凭什么这个子任务值得先做）。

这个仓库提供的不只是"更好的 prompt 模板"，而是：
- **一套诊断方法论**（EDOT：外部诊断驱动的在职培训）
- **全部实验代码**（408 次 Agent 运行，12 条件大横评）
- **可复用的路由配置与 Prompt 库**

---

## 2. 工作流架构

### D-E-A：注意力配置的三层漏斗

```
任务输入 → [D] 拆解为独立子任务 → [E] 为每个子任务赋优先级 → [A] 选择性配置注意力 → 产出
              ↑ 自反思能修               ↑ 自反思修不了               ↑
                                     需要外部诊断的跨实例视野
```

- **D（Decoupling）**：拆解。1 次类型转换。
- **E（Evaluation）**：评估优先级。**3 次类型转换**（定性→定量、绝对→相对、描述→指令）。这是瓶颈。
- **A（Aggregation）**：按优先级分配注意力，低价值直接放弃。1 次类型转换。

关键发现：**D:E:A 的类型转换比是 1:3:1。** E 需要的 3 次不可压缩的认知转换，决定了它永远是瓶颈。

### EDOT：外部诊断驱动的在职培训

```
一批任务跑完 → 外部强模型诊断（看全部）→ 人类审核方向 → 改进写入长期记忆 → 下批任务自动调用
```

**三个核心设计原则**（全部经实验验证）：
- 诊断**批量做**，不要实时做——实时诊断看不到跨实例的基准分布（+9pp）
- 诊断模型**必须比**执行模型强——否则退化为自反思（净效应为零）
- 自反思缓冲层**不要加**——纯浪费计算

### 无干预协议

人类操作者仅审核诊断方向的正确性，不介入分析内容或报告撰写。唯一的干预是事实纠错（如"明天见"在特定社区是浪漫主义而非反讽），依赖常识级别判断。

---

## 3. 核心指标

| 指标 | 数据 | 数值 |
|:--|:--|:--|
| **每日积分硬约束** | QClaw 调度预算 | 800 积分/日（DeepSeek 网页版免费） |
| **业务管线成本** | QClaw 日常运营 | ¥0/月 |
| **E/(D+A) 瓶颈比** | 12 条件大横评 | 基线 1.08 → 批量诊断 3.00 |
| **批量 vs 实时** | 2×2 分离矩阵 | 批量 92% vs 实时 83%（+9pp） |
| **更好的模型** | 2×2 分离矩阵 | Mimo 97% vs V4Pro 92%（+5pp） |
| **D 旗压缩率** | V1→V2 prompt 改进 | -91%（new36） |
| **E 旗抗压缩性** | 跨 5+ 数据集 | 始终 27~34，不受 prompt 影响 |
| **自动化拐点** | V2.0 后 5 轮 | 指标收敛，切换维护模式 |
| **总实验规模** | 全部实验 | 408 次 Agent 运行，228 条诊断记录 |

---

## 4. 快速开始

### 如果你是 AI 运营

读 `docs/preprint_link.md`（方法论摘要）+ `docs/cost_structure.md`（成本结构）。
核心操作：部署 Agent 系统后，每完成一批任务，用更强的模型做一次方法论审查，把改进写进长期记忆。诊断批量做，不要实时做。

### 如果你是 Agent 开发者

```bash
git clone https://github.com/muxueyoyo-bee/ai-cognitive-pipeline-edot.git
cd ai-cognitive-pipeline-edot
pip install openai datasets numpy
export DEEPSEEK_API_KEY="sk-xxx"
python experiments/agent_system.py          # 三阶段流水线
python experiments/grand_review.py          # 12 条件大横评
```

核心脚本见 `experiments/README.md`。在你的系统中加入：(1) 长期记忆模块，(2) 外部诊断接口，(3) 批量诊断触发机制。路由配置参考 `configs/agent_routing.json`。

### Prompt 模板

`configs/prompt_templates/` 下提供 D-E-A 各环节 + 批量诊断 + 记忆注入的标准 Prompt 模板，含占位符与调用说明，可直接复用。

---

## 5. 仓库结构

```
ai-cognitive-pipeline-edot/
├── README.md                          # 主文档
├── LICENSE                            # MIT（代码）
├── NOTICE.md                          # 分层许可 + 版权声明
├── .gitignore
├── docs/                              # 方法论与边界文档
│   ├── preprint_link.md               # 预印本摘要 + DOI
│   ├── cost_structure.md              # ¥0 业务 / ¥70 学术成本隔离
│   └── plateau_analysis.md            # 自动化拐点与切换 SOP
├── experiments/                       # 全部实验代码（16 脚本）
├── configs/                           # 路由配置 + Prompt 模板库
│   ├── agent_routing.json             # 执行-诊断解耦路由
│   ├── memory_injection.yaml          # 长期记忆注入规范
│   └── prompt_templates/              # D-E-A / EDOT 标准 Prompt 库
├── logs/                              # 可审计运行日志
│   ├── iteration_summary.csv          # 22 轮诊断关键节点
│   └── v1_to_v2_metrics.json          # V1.0→V2.0 指标对比
├── reports/                           # 运营报告输出（脱敏版）
│   ├── v1.0_baseline/
│   ├── v1.5_edot_cycle/
│   ├── v2.0_stable/
│   └── genshin_cross_game/
├── src/                               # 核心代码与工具链
│   ├── qclaw_pipeline/                # QClaw 分析管线
│   ├── social_data_cli/               # 社交媒体分析 CLI
│   └── honkai_data_viz/               # 崩铁养成曲线
└── assets/                            # 演示素材（外链）
```

---

## 项目边界声明

这是我（一名大二经济学学生）的独立研究项目。它不是学术论文——没有同行评审，没有机构背书，没有导师指导。它是 15 天高强度工程实践的沉淀：从搭建 Multi-Agent 系统跑出第一批运营报告，到设计 SWE-bench 受控实验发现 D-E-A 瓶颈，到形式化 Σα_i=1 约束推导 λ 门槛效应。

你可以在简历里引用它，在项目里复用它的方法论，在面试里讲它的故事。但别说这是"发表"——说这是"做了个实验觉得有意思，写了报告"。

---

## 许可证

| 资产类型 | 许可 | 核心条款 |
|:--|:--|:--|
| 代码/脚本 (`experiments/`, `src/`, `configs/`) | MIT | 自由使用、修改、商用，须保留版权声明 |
| 文档 (`docs/`) | CC BY-NC-SA 4.0 | 学习/分享/衍生，非商用 + 相同方式共享 |
| 日志/报告 (`logs/`, `reports/`) | CC BY 4.0 | 任何使用，须署名 |
| 素材 (`assets/`) | CC BY-NC 4.0 | 非商用展示 |

详见 [NOTICE.md](NOTICE.md) 和 [LICENSE](LICENSE)。

---

© 2026 龙文凭 (Long Wenping) | 南京邮电大学 | 独立架构验证
