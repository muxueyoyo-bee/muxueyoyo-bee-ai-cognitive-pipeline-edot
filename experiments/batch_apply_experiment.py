"""
V4Pro批量诊断应用实验 —— 填补2x2分离矩阵的V4Pro Batch单元格。
将V4Pro批量诊断的系统性发现写入V2 prompts, 重跑全部72实例。
"""
import json, os, sys, re
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent_system import DeepSeekLLM, KnowledgeBase, AgentPrompts, DEAPipeline

BASE = Path("D:/数据/论文数据/SWE-bench_DEA")
OUT = BASE / "批量诊断_V4Pro"
OUT.mkdir(parents=True, exist_ok=True)

# Load V2 prompts
with open(BASE / "prompts_v2_for_mimo_review.json", "r", encoding="utf-8") as f:
    v2 = json.load(f)

# Extract batch diagnosis systemic findings
batch_md = open(BASE / "separation_experiment/batch_v4pro/batch_diagnosis.md", "r", encoding="utf-8").read()

# Build batch-improved prompts by appending systemic feedback
batch_feedback = """
## V4Pro 批量诊断反馈 (72条轨迹D-E-A审查)

### 核心发现
1. **E_NO_JUSTIFICATION 是压倒性的剩余缺陷** (44/60实例): Fixer标注了P0/P1/P2标签但系统性地跳过了论证步骤。
2. **E_NO_ABANDON** (6实例): 部分子问题超出issue范围, 应被放弃但未被标记[ABANDONED]。
3. **Reviewer早期审查不严格** : 多例中Reviewer接受了含有明显缺陷的patch, 后期修订轮次耗尽后才REJECT。
4. **old36 D_MONOLITH** (10次): astropy复杂issue仍被作为单一问题处理。

### 强制改进要求
- **P0标签必须附带至少一句论证**: 格式为"P0理由: <impact/dependency/effort>"
- **子问题>3时, 必须标记至少一个[ABANDONED]**: 说明放弃理由
- **Reviewer在A0时必须执行严格审查**: 不能等到修订轮次耗尽才REJECT
- **每个子问题的优先级必须互不相同**: 允许P0/P1/P2各一个, 禁止全部标为P0
"""

batch_improved_locator = v2["locator"] + "\n\n" + batch_feedback
batch_improved_fixer = v2["fixer"] + "\n\n" + batch_feedback
batch_improved_reviewer = v2["reviewer"] + "\n\n" + batch_feedback

prompts = AgentPrompts(
    version="v4.0_batch_v4pro",
    locator=batch_improved_locator,
    fixer=batch_improved_fixer,
    reviewer=batch_improved_reviewer,
)

# Load instances
from datasets import load_from_disk
ds = load_from_disk("D:/数据/论文数据/SWE-bench_Verified")
all_instances = list(ds)
old36 = all_instances[:36]
new36 = all_instances[36:72]
for inst in old36:
    inst["_set"] = "old36"
for inst in new36:
    inst["_set"] = "new36"
instances = old36 + new36

print(f"Loaded {len(old36)} old + {len(new36)} new = {len(instances)} total")

# Run
exec_llm = DeepSeekLLM()
kb = KnowledgeBase(persistent_file=str(OUT / "kb.json"))

print(f"\n{'='*70}")
print(f"V4Pro批量诊断应用实验 (n=72)")
print(f"  Prompts: v4.0_batch_v4pro (V2 + V4Pro批量反馈)")
print(f"  Output: {OUT}")
print(f"{'='*70}\n")

pipeline = DEAPipeline(exec_llm, prompts, kb)

results = []
for i, inst in enumerate(instances):
    inst_id = inst["instance_id"]
    inst_set = inst.get("_set", "?")
    print(f"[批] {i+1}/{len(instances)}", end="")
    try:
        result = pipeline.run_one(inst)
        result["global_index"] = i
        result["_set"] = inst_set
        result["prompt_version"] = "v4.0_batch_v4pro"
        results.append(result)

        traj_file = OUT / f"{inst_id.replace('/', '_')}.json"
        with open(traj_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if (i + 1) % 10 == 0:
            kb.save(str(OUT / "kb.json"))
            with open(OUT / "checkpoint.json", "w") as f:
                json.dump({"last_idx": i, "n_results": len(results)}, f)
            print(f"  [checkpoint @ {i+1}]")
    except Exception as e:
        print(f"  !! Failed: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "instance_id": inst_id, "_set": inst_set, "global_index": i,
            "error": str(e), "flags": ["EXEC_ERROR"], "reviewer_verdict": "ERROR"
        })

# Save
kb.save(str(OUT / "kb.json"))
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump([{k: v for k, v in r.items() if k != "trajectory"} for r in results],
              f, ensure_ascii=False, indent=2)

# Quick stats
valid = [r for r in results if not r.get("error")]
passes = sum(1 for r in valid if r.get("reviewer_verdict") == "PASS")
flags = [f for r in valid for f in r.get("flags", [])]
d = sum(1 for f in flags if f.startswith("D_"))
e = sum(1 for f in flags if f.startswith("E_"))
a = sum(1 for f in flags if f.startswith("A_"))
er = e / max(d + a, 1)
print(f"\n{'='*70}")
print(f"V4Pro批量应用实验完成")
print(f"  PASS: {passes}/{len(valid)} ({passes/max(len(valid),1):.0%})")
print(f"  D={d}, E={e}, A={a}, E/(D+A)={er:.2f}")
print(f"  Output: {OUT}")
print(f"{'='*70}")
