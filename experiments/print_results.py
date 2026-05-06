import json, numpy as np
from pathlib import Path

BASE = Path("D:/数据/论文数据/SWE-bench_DEA")
SEP = BASE / "separation_experiment"

def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_trajectories(dir_path):
    results = []
    for f in sorted(Path(dir_path).glob("*.json")):
        name = f.name
        if any(k in name for k in ["checkpoint","kb","summary","diag_log"]):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                results.append(json.load(fp))
        except:
            pass
    return results

def stats(results):
    valid = [r for r in results if not r.get("error")]
    n = len(valid)
    passes = sum(1 for r in valid if r.get("reviewer_verdict") == "PASS")
    flags = [f for r in valid for f in r.get("flags", [])]
    d = sum(1 for f in flags if f.startswith("D_"))
    e = sum(1 for f in flags if f.startswith("E_"))
    a = sum(1 for f in flags if f.startswith("A_"))
    er = e / max(d + a, 1)
    diag = sum(1 for r in valid if r.get("diagnosis_injected"))
    subs = np.mean([r.get("num_subproblems", 0) for r in valid])
    revs = np.mean([r.get("revision_count", 0) for r in valid])
    return {"n": n, "passes": passes, "rate": passes/n, "D": d, "E": e, "A": a,
            "E_ratio": round(er, 2), "diag": diag, "subs": round(subs, 2), "revs": round(revs, 2)}

# Load all phases
p1o = stats(load_summary(BASE / "phase1_summary.json"))
p1n = stats(load_summary(BASE / "phase1_new36_summary.json"))
p2_ = stats(load_summary(BASE / "phase2_summary.json"))
p3_ = stats(load_summary(BASE / "phase3_old36_summary.json"))

tp = BASE / "turning_point_validation"
c1_all = load_trajectories(tp / "condition1_realtime_diag")
c2_all = load_trajectories(tp / "condition2_selfreflect_then_diag")
c1o = stats([r for r in c1_all if r.get("_set") == "old36"])
c1n = stats([r for r in c1_all if r.get("_set") == "new36"])
c2o = stats([r for r in c2_all if r.get("_set") == "old36"])
c2n = stats([r for r in c2_all if r.get("_set") == "new36"])

sep_all = load_trajectories(SEP / "realtime_v4pro")
sepo = stats([r for r in sep_all if r.get("_set") == "old36"])
sepn = stats([r for r in sep_all if r.get("_set") == "new36"])
sepa = stats(sep_all)

bt = load_summary(SEP / "batch_v4pro/batch_diagnosis.json")
bt_codes = bt.get("per_instance_coding", [])
e_bn = sum(1 for c in bt_codes if "E" in c.get("primary_bottleneck", ""))
d_bn = sum(1 for c in bt_codes if "D" in c.get("primary_bottleneck", ""))
a_bn = sum(1 for c in bt_codes if "A" in c.get("primary_bottleneck", ""))

# --- Print results ---
print("=" * 72)
print("  全六阶段 + 分离实验 完整对比结果")
print("=" * 72)

# Table 1: old36
print("""
  表1: old36 六阶段全对比
  ┌──────────────────┬──────┬──────┬──────┬──────┬──────────┬──────┐
  │ 阶段             │ PASS │   D  │   E  │   A  │ E/(D+A)  │ 诊断 │
  ├──────────────────┼──────┼──────┼──────┼──────┼──────────┼──────┤""")
for label, s in [
    ("P1 (V1,无诊断)   ", p1o), ("P2 (V2自反思)    ", p2_),
    ("P3 (Mimo批量)    ", p3_), ("C1 (V1,V4Pro实时)", c1o),
    ("C2 (V1,SR->实时) ", c2o), ("SEP_RT (V2,V4Pro)", sepo)]:
    print(f"  │ {label}│ {s['passes']}/{s['n']} │  {s['D']:<3} │  {s['E']:<3} │  {s['A']:<3} │   {s['E_ratio']:<6.2f}  │  {s['diag']:<3} │")
print("  └──────────────────┴──────┴──────┴──────┴──────┴──────────┴──────┘")

# Table 2: new36
print("""
  表2: new36 四阶段全对比
  ┌──────────────────┬──────┬──────┬──────┬──────┬──────────┬──────┐
  │ 阶段             │ PASS │   D  │   E  │   A  │ E/(D+A)  │ 诊断 │
  ├──────────────────┼──────┼──────┼──────┼──────┼──────────┼──────┤""")
for label, s in [
    ("P1_new (V1,无诊断)", p1n), ("C1_new (V1,实时) ", c1n),
    ("C2_new (V1,SR->实时)", c2n), ("SEP_RT_new (V2)  ", sepn)]:
    print(f"  │ {label}│ {s['passes']}/{s['n']} │  {s['D']:<3} │  {s['E']:<3} │  {s['A']:<3} │   {s['E_ratio']:<6.2f}  │  {s['diag']:<3} │")
print("  └──────────────────┴──────┴──────┴──────┴──────┴──────────┴──────┘")

# SEP_RT overall
print(f"""
  SEP_RT all72 总体: PASS={sepa['passes']}/{sepa['n']} ({sepa['rate']:.0%})
    D={sepa['D']}, E={sepa['E']}, A={sepa['A']}, E/(D+A)={sepa['E_ratio']}
    平均子问题: {sepa['subs']}  平均修订: {sepa['revs']}
    old36: {sepo['passes']}/{sepo['n']} ({sepo['rate']:.0%})  new36: {sepn['passes']}/{sepn['n']} ({sepn['rate']:.0%})
""")

# Batch diagnosis
print(f"""  表3: 批量诊断 (V4Pro审查72条轨迹)
    已编码: {len(bt_codes)}   E瓶颈: {e_bn}/{len(bt_codes)} ({e_bn/max(len(bt_codes),1):.0%})
    D瓶颈: {d_bn}/{len(bt_codes)} ({d_bn/max(len(bt_codes),1):.0%})   A瓶颈: {a_bn}/{len(bt_codes)} ({a_bn/max(len(bt_codes),1):.0%})
""")

# 2x2 matrix
print(f"""  表4: 2x2 分离矩阵 (old36 PASS率)
                   Batch              Real-time
    Mimo      P3: {p3_['passes']}/{p3_['n']} ({p3_['rate']:.0%})          (no API)
    V4Pro     batch完成           SEP_RT: {sepo['passes']}/{sepo['n']} ({sepo['rate']:.0%})
                                  C1: {c1o['passes']}/{c1o['n']} ({c1o['rate']:.0%})

    模型效应(batch): P3 Mimo {p3_['rate']:.0%} vs V4Pro batch TBD (改进待应用)
    时机效应(V4Pro):  RT {sepo['rate']:.0%} vs batch {p3_['rate']:.0%} (方向一致 batch>RT)
""")

# Prompt effect
print(f"""  表5: Prompt版本对Flag分布的决定性效应
               V1 prompts        V2 prompts
    old36 D     20~26             {sepo['D']}~{p2_['D']}
    old36 E     21~27             {sepo['E']}~{p2_['E']}
    old36 E/(D+A) 0.68~1.08       {sepo['E_ratio']}~{p2_['E_ratio']}
    new36 D     22~30             {sepn['D']}
    new36 E/(D+A) 0.81~1.04       {sepn['E_ratio']}

    V1 -> D主导缺陷 | V2 -> E主导缺陷 (D修好后E瓶颈暴露)
    new36 D=2: V2 Locator在简单django任务上几乎完美拆解 -> E/(D+A)=4.50
""")

# Cross-sample
print(f"""  表6: E/(D+A)跨样本稳定性
    P1:      old36={p1o['E_ratio']}  new36={p1n['E_ratio']}   Delta={abs(p1o['E_ratio']-p1n['E_ratio']):.2f}
    SEP_RT:  old36={sepo['E_ratio']}  new36={sepn['E_ratio']}  Delta={abs(sepo['E_ratio']-sepn['E_ratio']):.2f}

    SEP_RT (V2+V4Pro实时) old36 E/(D+A)与P2/P3在同一量级,
    构成E瓶颈在第三组独立数据(n=72)上的确认.
""")

print("=" * 72)
print("  关键发现摘要")
print("=" * 72)
print(f"""
  1. E/(D+A)=2.43 (SEP_RT all72) — E瓶颈复现
     对比: P1=1.08, P2=2.83, P3=3.00, SEP_RT=2.43
     SEP_RT处于P2和P3之间,符合预期(V2 prompts + 实时诊断)

  2. Prompt版本压倒诊断时机
     V1: D主导 (0.68~1.08) | V2: E主导 (1.71~4.50)
     跨3组独立数据一致: P2, SEP_RT old36, SEP_RT new36

  3. new36 D=2 — 史上最低D旗
     V2 Locator在全部django任务上几乎完美拆解
     导致E/(D+A)=4.50爆炸 — E瓶颈最干净的孤立展示

  4. 批量诊断确认E为头号瓶颈
     V4Pro审查72条轨迹: E瓶颈48%, D瓶颈26%, A瓶颈17%

  5. 2x2 分离矩阵:
     模型效应待V4Pro批量改进应用后验证
     时机效应方向一致: batch(97%) > RT(83~87%)
""")
