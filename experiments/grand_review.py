"""
Grand Cross-Review: all phases × all conditions × all metrics.
Single authoritative table for the four-paper system.
"""
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
        except: pass
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
    return {"n": n, "PASS": f"{passes}/{n}", "rate": passes/n, "D": d, "E": e, "A": a,
            "E/(D+A)": round(er, 2), "diag": diag, "subs": round(subs, 2), "revs": round(revs, 2)}

# Load everything
p1o = stats(load_summary(BASE / "phase1_summary.json"))
p1n = stats(load_summary(BASE / "phase1_new36_summary.json"))
p2_ = stats(load_summary(BASE / "phase2_summary.json"))
p3_ = stats(load_summary(BASE / "phase3_old36_summary.json"))

tp = BASE / "turning_point_validation"
c1_all = load_trajectories(tp / "condition1_realtime_diag")
c2_all = load_trajectories(tp / "condition2_selfreflect_then_diag")
c1o = stats([r for r in c1_all if r.get("_set") == "old36"])
c1n = stats([r for r in c1_all if r.get("_set") == "new36"])
c1a = stats(c1_all)
c2o = stats([r for r in c2_all if r.get("_set") == "old36"])
c2n = stats([r for r in c2_all if r.get("_set") == "new36"])
c2a = stats(c2_all)

sep_all = load_trajectories(SEP / "realtime_v4pro")
sepo = stats([r for r in sep_all if r.get("_set") == "old36"])
sepn = stats([r for r in sep_all if r.get("_set") == "new36"])
sepa = stats(sep_all)

bt = load_summary(SEP / "batch_v4pro/batch_diagnosis.json")
bt_codes = bt.get("per_instance_coding", [])
e_bn = sum(1 for c in bt_codes if "E" in c.get("primary_bottleneck", ""))
d_bn = sum(1 for c in bt_codes if "D" in c.get("primary_bottleneck", ""))
a_bn = sum(1 for c in bt_codes if "A" in c.get("primary_bottleneck", ""))

print("=" * 120)
print("  大 横 评 —— 全 阶 段 × 全 条 件 × 全 指 标")
print("=" * 120)

# ===== TABLE 1: Complete Phase Matrix =====
print("""
  表1: 全阶段横向对比矩阵
  ┌──────────────────────────┬──────┬──────┬────┬────┬────┬──────────┬──────┬──────┬──────┐
  │ 条件                     │ 实例 │ PASS │ D  │ E  │ A  │ E/(D+A)  │ 诊断 │ 子问 │ 修订 │
  ├──────────────────────────┼──────┼──────┼────┼────┼────┼──────────┼──────┼──────┼──────┤""")

rows = [
    ("P1 (V1,无诊断)       old36", p1o),
    ("P1_new (V1,无诊断)   new36", p1n),
    ("P2 (V2自反思,无诊断) old36", p2_),
    ("P3 (Mimo批量,V3)     old36", p3_),
    ("C1 (V1,V4Pro实时)    old36", c1o),
    ("C1_new (V1,V4Pro实时) new36", c1n),
    ("C2 (V1,SR->V4Pro)    old36", c2o),
    ("C2_new (V1,SR->V4Pro) new36", c2n),
    ("SEP_RT (V2,V4Pro实时) old36", sepo),
    ("SEP_RT_new (V2,V4Pro) new36", sepn),
    ("SEP_RT (V2,V4Pro实时) all72", sepa),
]
for label, s in rows:
    print(f"  │ {label:<24} │  {s['n']:>3} │ {s['PASS']:>4} │ {s['D']:>2} │ {s['E']:>2} │ {s['A']:>2} │   {s['E/(D+A)']:>6.2f}  │  {s['diag']:>3} │ {s['subs']:>4} │ {s['revs']:>4} │")
print("  └──────────────────────────┴──────┴──────┴────┴────┴────┴──────────┴──────┴──────┴──────┘")

# ===== TABLE 2: Prompt Version Effect =====
print("""
  表2: Prompt版本效应 —— V1 vs V2 的分层对比
  ┌──────────────────────┬──────────┬──────────┬──────────┐
  │ 指标                 │ V1 (基线)│ V2 (自反思改进)│ Δ       │
  ├──────────────────────┼──────────┼──────────┼──────────┤""")
# V1: P1_old, C1_old, C2_old / V2: P2, SEP_RT_old
v1_d_range = f"{min(c2o['D'],c1o['D'],p1o['D'])}~{max(c2o['D'],c1o['D'],p1o['D'])}"
v2_d_range = f"{min(p2_['D'],sepo['D'])}~{max(p2_['D'],sepo['D'])}"
v1_e_range = f"{min(c2o['E'],c1o['E'],p1o['E'])}~{max(c2o['E'],c1o['E'],p1o['E'])}"
v2_e_range = f"{min(p2_['E'],sepo['E'])}~{max(p2_['E'],sepo['E'])}"
v1_er = f"{min(c2o['E/(D+A)'],c1o['E/(D+A)'],p1o['E/(D+A)']):.2f}~{max(c2o['E/(D+A)'],c1o['E/(D+A)'],p1o['E/(D+A)']):.2f}"
v2_er = f"{min(p2_['E/(D+A)'],sepo['E/(D+A)']):.2f}~{max(p2_['E/(D+A)'],sepo['E/(D+A)']):.2f}"
v1_pass = f"{min(c2o['rate'],c1o['rate'],p1o['rate']):.0%}~{max(c2o['rate'],c1o['rate'],p1o['rate']):.0%}"
v2_pass = f"{min(p2_['rate'],sepo['rate']):.0%}~{max(p2_['rate'],sepo['rate']):.0%}"

print(f"  │ old36 D旗             │ {v1_d_range:<8} │ {v2_d_range:<8} │ {'-55%':<8} │")
print(f"  │ old36 E旗             │ {v1_e_range:<8} │ {v2_e_range:<8} │ {'+26%~+7%':<8} │")
print(f"  │ old36 E/(D+A)         │ {v1_er:<8} │ {v2_er:<8} │ {'+60%~+180%':<8} │")
print(f"  │ old36 PASS率          │ {v1_pass:<8} │ {v2_pass:<8} │ {'-3~+6pp':<8} │")
print(f"  │ new36 D旗             │ {min(c2n['D'],c1n['D'],p1n['D'])}~{max(c2n['D'],c1n['D'],p1n['D']):<6} │ {sepn['D']:<8} │ {'-91%':<8} │")
print(f"  │ new36 E/(D+A)         │ {min(c2n['E/(D+A)'],c1n['E/(D+A)'],p1n['E/(D+A)']):.2f}~{max(c2n['E/(D+A)'],c1n['E/(D+A)'],p1n['E/(D+A)']):.2f} │ {sepn['E/(D+A)']:<8} │ {'+330%':<8} │")
print("  └──────────────────────┴──────────┴──────────┴──────────┘")

# ===== TABLE 3: 2x2 Separation Matrix =====
print(f"""
  表3: 2×2 分离矩阵 (old36 PASS率)
  ┌──────────┬──────────────────────┬──────────────────────┐
  │          │ 批量                 │ 实时                 │
  ├──────────┼──────────────────────┼──────────────────────┤
  │ Mimo     │ P3: {p3_['PASS']} ({p3_['rate']:.0%})        │ (no API)             │
  ├──────────┼──────────────────────┼──────────────────────┤
  │ V4Pro    │ SEP_BT: 审查完成     │ SEP_RT: {sepo['PASS']} ({sepo['rate']:.0%})    │
  │          │ (改进待应用)         │ C1: {c1o['PASS']} ({c1o['rate']:.0%})          │
  └──────────┴──────────────────────┴──────────────────────┘

  模型效应 (batch): P3 Mimo {p3_['rate']:.0%} vs V4Pro batch — 待V4Pro批量改进应用后验证
                    Mimo 2.5 Pro (~Opus 4.7) vs V4Pro — 能力差距方向已知

  时机效应 (V4Pro): RT {sepo['rate']:.0%}~{c1o['rate']:.0%} vs batch P3 {p3_['rate']:.0%} — 方向一致 batch>RT
                    差距 {p3_['rate']-sepo['rate']:.0%}~{p3_['rate']-c1o['rate']:.0%}pp (混杂了prompt版本差异)

  Prompt效应 (old36): V1 RT {c1o['rate']:.0%}~{c2o['rate']:.0%} vs V2 RT {sepo['rate']:.0%}
                     V2的Reviewer更严格→更多REJECT→PASS率略低但缺陷检测更完整
""")

# ===== TABLE 4: E/(D+A) Cross-Sample Stability =====
print(f"""  表4: E/(D+A) 跨样本稳定性
  ┌────────────┬──────────┬──────────┬───────┬─────────────────────┐
  │ 条件        │ old36    │ new36    │ Δ     │ 解读                │
  ├────────────┼──────────┼──────────┼───────┼─────────────────────┤
  │ P1 (V1)    │ {p1o['E/(D+A)']:<8.2f} │ {p1n['E/(D+A)']:<8.2f} │ {abs(p1o['E/(D+A)']-p1n['E/(D+A)']):.2f}   │ 跨样本一致, p=0.91   │
  │ SEP_RT (V2)│ {sepo['E/(D+A)']:<8.2f} │ {sepn['E/(D+A)']:<8.2f} │ {abs(sepo['E/(D+A)']-sepn['E/(D+A)']):.2f}  │ D旗差异导致比率分化  │
  └────────────┴──────────┴──────────┴───────┴─────────────────────┘

  P1: E/(D+A)跨样本稳定 (D旗都在20~22, 任务复杂度差异未影响比率)
  SEP_RT: old36=1.71 (D=10), new36=4.50 (D=2) — 不是跨样本不稳定,
  而是V2在不同任务复杂度上对D的压缩效果不同 (astropy复杂→D仍10,
  django简单→D仅2), 导致分母差异放大。但E旗绝对密度在两样本上
  一致 (29 vs 27)——E瓶颈本身是跨样本稳定的。
""")

# ===== TABLE 5: Batch Diagnosis D-E-A Coding =====
print(f"""  表5: V4Pro 批量诊断 D-E-A 审查 (72条轨迹)
  ┌──────────┬──────┬──────────────────────────────────────┐
  │ 主要瓶颈 │ 数量 │ 关键发现                             │
  ├──────────┼──────┼──────────────────────────────────────┤
  │ E        │ {e_bn}/{len(bt_codes)} │ E_NO_JUSTIFICATION跨repo一致            │
  │          │({e_bn/max(len(bt_codes),1):.0%})│ "evaluation step is systematically weak" │
  │ D        │ {d_bn}/{len(bt_codes)} │ old36 D_MONOLITH 10次, new36仅2次       │
  │          │({d_bn/max(len(bt_codes),1):.0%})│ "Locator has learned to decompose"       │
  │ A        │ {a_bn}/{len(bt_codes)} │ Reviewer早期不严格,后期耗尽修订轮次     │
  │          │({a_bn/max(len(bt_codes),1):.0%})│ 5个A_FALSE_PASS漏判                    │
  └──────────┴──────┴──────────────────────────────────────┘
""")

# ===== TABLE 6: Evidence Convergence =====
print("""  表6: E瓶颈的证据汇聚 (六条独立证据线)
  ┌────┬────────────────────────────────┬────────────────────────────────┐
  │ #  │ 证据                          │ 核心发现                       │
  ├────┼────────────────────────────────┼────────────────────────────────┤""")
evidences = [
    ("1","类型转换比独立论证","D:E:A = 1:3:1, 从输入输出类型独立推导"),
    ("2","自反思实验 (P1→P2)","D可自修复(-55%), E不可(+26%)"),
    ("3","外部诊断实验 (P2→P3)","Mimo介入: +5pp, E/(D+A)→3.00"),
    ("4","72实例时机验证 (C1/C2)","批量>实时 11.1pp, SR缓冲层净效应=0"),
    ("5","分离实验 (SEP_RT, n=72)","E/(D+A)=2.43, CI[1.66,3.82]不含1.0"),
    ("6","Prompt版本效应 (V1 vs V2)","V2压缩D 55~91%, E密度不变(27~34)"),
]
for num, title, finding in evidences:
    print(f"  │ {num}  │ {title:<30} │ {finding:<30} │")
print("  └────┴────────────────────────────────┴────────────────────────────────┘")

# ===== Summary =====
n_total_agents = 36*3 + 72*2 + 72 + 6*2  # P1+P2+P3 + C1+C2 + SEP_RT + TP
print(f"""
  ===== 实验规模总览 =====
  独立Agent运行: ~{n_total_agents}次 (P1+P2+P3=108, C1+C2=144, SEP_RT=72, TP=12)
  诊断记录:      ~216条 (P3批量36 + C1实时72 + C2实时72 + SEP_RT实时30 + SEP_BT批量42)
  PASS率变化:    83% → 97% (old36, 跨六阶段全范围)
  E/(D+A)变化:   0.68 → 4.50 (全条件全范围)
  D旗变化:       30 → 2 (V1 new36 → V2 new36, 跨prompt全范围)

  最稳健发现 (跨5+独立数据集复现):
  1. E瓶颈始终存在 — E/(D+A)在V2/V3下始终 >1.0, CI不含1.0
  2. D可被prompt改进系统性压缩 — V1→V2 D旗降幅55~91%
  3. E不可被prompt改进压缩 — E旗密度始终27~34, 不受prompt版本/任务复杂度影响
  4. 批量诊断 > 实时诊断 — 方向一致, 差距11~14pp (old36)
""")
