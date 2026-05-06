"""Bootstrap CI for E/(D+A) ratio across phases, using instance-level resampling."""
import json
import numpy as np
from pathlib import Path

BASE = Path("D:/数据/论文数据/SWE-bench_DEA")

def load_flags(d: Path) -> dict[str, list[str]]:
    """Load per-instance flag lists."""
    flags = {}
    if not d.exists():
        return flags
    for f in d.glob("*.json"):
        if any(x in f.name for x in ["checkpoint", "kb", "summary", "diag_log"]):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            iid = data.get("instance_id", "")
            if iid:
                flags[iid] = data.get("flags", [])
        except:
            pass
    return flags

# Load all phases
p1 = load_flags(BASE / "phase1_baseline")
p2 = load_flags(BASE / "phase2_self_reflection")
p3 = load_flags(BASE / "phase3_old36")

# Only use instances present in all three phases
common_ids = sorted(set(p1.keys()) & set(p2.keys()) & set(p3.keys()))
print(f"Common instances across P1/P2/P3: {len(common_ids)}")

def compute_eda(flags_dict: dict, ids: list[str]) -> float:
    d = sum(1 for iid in ids for f in flags_dict.get(iid, []) if f.startswith("D_"))
    e = sum(1 for iid in ids for f in flags_dict.get(iid, []) if f.startswith("E_"))
    a = sum(1 for iid in ids for f in flags_dict.get(iid, []) if f.startswith("A_"))
    return e / max(d + a, 1)

# Point estimates
eda_p1 = compute_eda(p1, common_ids)
eda_p2 = compute_eda(p2, common_ids)
eda_p3 = compute_eda(p3, common_ids)
print(f"\nPoint estimates:")
print(f"  P1 E/(D+A) = {eda_p1:.2f}")
print(f"  P2 E/(D+A) = {eda_p2:.2f}")
print(f"  P3 E/(D+A) = {eda_p3:.2f}")

# Bootstrap CIs
np.random.seed(42)
n_boot = 10000
n = len(common_ids)

boot_p1 = []
boot_p2 = []
boot_p3 = []
boot_p1p2 = []  # P2 - P1
boot_p2p3 = []  # P3 - P2
boot_p1p3 = []  # P3 - P1

for _ in range(n_boot):
    idx = np.random.choice(n, size=n, replace=True)
    ids_boot = [common_ids[i] for i in idx]
    v1 = compute_eda(p1, ids_boot)
    v2 = compute_eda(p2, ids_boot)
    v3 = compute_eda(p3, ids_boot)
    boot_p1.append(v1)
    boot_p2.append(v2)
    boot_p3.append(v3)
    boot_p1p2.append(v2 - v1)
    boot_p2p3.append(v3 - v2)
    boot_p1p3.append(v3 - v1)

def ci95(vals):
    return np.percentile(vals, [2.5, 97.5])

print(f"\nBootstrap 95% CIs (10,000 resamples, instance-level):")
print(f"  P1 E/(D+A): {eda_p1:.2f}  CI=[{ci95(boot_p1)[0]:.2f}, {ci95(boot_p1)[1]:.2f}]")
print(f"  P2 E/(D+A): {eda_p2:.2f}  CI=[{ci95(boot_p2)[0]:.2f}, {ci95(boot_p2)[1]:.2f}]")
print(f"  P3 E/(D+A): {eda_p3:.2f}  CI=[{ci95(boot_p3)[0]:.2f}, {ci95(boot_p3)[1]:.2f}]")

print(f"\n  Δ(P2-P1): {eda_p2-eda_p1:+.2f}  CI=[{ci95(boot_p1p2)[0]:+.2f}, {ci95(boot_p1p2)[1]:+.2f}]")
print(f"  Δ(P3-P2): {eda_p3-eda_p2:+.2f}  CI=[{ci95(boot_p2p3)[0]:+.2f}, {ci95(boot_p2p3)[1]:+.2f}]")
print(f"  Δ(P3-P1): {eda_p3-eda_p1:+.2f}  CI=[{ci95(boot_p1p3)[0]:+.2f}, {ci95(boot_p1p3)[1]:+.2f}]")

# Also check P3 sensitivity to +1 A flag
d3 = sum(1 for iid in common_ids for f in p3.get(iid, []) if f.startswith("D_"))
e3 = sum(1 for iid in common_ids for f in p3.get(iid, []) if f.startswith("E_"))
a3 = sum(1 for iid in common_ids for f in p3.get(iid, []) if f.startswith("A_"))
print(f"\nP3 flag composition: D={d3} E={e3} A={a3}")
print(f"  E/(D+A) = {e3}/{d3+a3} = {e3/(d3+a3):.2f}")
print(f"  If +1 A flag: E/(D+A+1) = {e3}/{d3+a3+1} = {e3/(d3+a3+1):.2f}")
print(f"  If -1 A flag: E/(D+A-1) = {e3}/{d3+a3-1} = {e3/(d3+a3-1):.2f}")
print(f"  If +1 D flag: E/(D+1+A) = {e3}/{d3+a3+1} = {e3/(d3+a3+1):.2f}")
print(f"  Sensitivity range: [{e3/(d3+a3+1):.2f}, {e3/max(d3+a3-1,1):.2f}]")
