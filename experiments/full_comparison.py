"""Full cross-phase comparison: P1 → P2 → P3 → C1 → C2"""
import json
from pathlib import Path

BASE = Path("D:/数据/论文数据/SWE-bench_DEA")

def load_dir(d: Path) -> dict:
    results = {}
    if not d.exists():
        return results
    for f in d.glob("*.json"):
        if any(x in f.name for x in ["checkpoint", "kb", "summary", "diag_log"]):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            iid = data.get("instance_id", "")
            if iid:
                v = data.get("final_verdict") or data.get("reviewer_verdict", "?")
                results[iid] = {
                    "verdict": v,
                    "flags": data.get("flags", []),
                    "subs": data.get("num_subproblems", 0),
                    "revisions": data.get("revision_count", 0),
                }
        except:
            pass
    return results

def load_summary(fpath: Path) -> dict:
    results = {}
    if not fpath.exists():
        return results
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                iid = item.get("instance_id", "")
                if iid:
                    v = item.get("final_verdict") or item.get("reviewer_verdict", "?")
                    results[iid] = {
                        "verdict": v,
                        "flags": item.get("flags", []),
                        "subs": item.get("num_subproblems", 0),
                        "revisions": item.get("revision_count", 0),
                    }
        elif isinstance(data, dict):
            for iid, item in data.items():
                if isinstance(item, dict):
                    v = item.get("final_verdict") or item.get("reviewer_verdict", "?")
                    results[iid] = {"verdict": v}
    except:
        pass
    return results

# Load all phases
p1_old = load_dir(BASE / "phase1_baseline")
p1_new = load_summary(BASE / "phase1_new36_summary.json")
p2 = load_dir(BASE / "phase2_self_reflection")
p3 = load_dir(BASE / "phase3_old36")
c1 = load_dir(BASE / "turning_point_validation/condition1_realtime_diag")
c2 = load_dir(BASE / "turning_point_validation/condition2_selfreflect_then_diag")

# Load C1/C2 diag info
c1_diag = set()
c2_diag = set()
c2_sr = set()
for f in (BASE / "turning_point_validation/condition1_realtime_diag").glob("*.json"):
    if any(x in f.name for x in ["checkpoint", "kb", "summary", "diag_log"]):
        continue
    try:
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        if d.get("diagnosis_injected"):
            c1_diag.add(d["instance_id"])
    except: pass
for f in (BASE / "turning_point_validation/condition2_selfreflect_then_diag").glob("*.json"):
    if any(x in f.name for x in ["checkpoint", "kb", "summary", "diag_log"]):
        continue
    try:
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        if d.get("diagnosis_injected"):
            c2_diag.add(d["instance_id"])
        if d.get("self_reflection_done"):
            c2_sr.add(d["instance_id"])
    except: pass

# Collect all instance IDs
all_ids = set()
all_ids.update(p1_old.keys(), p1_new.keys(), p2.keys(), p3.keys(), c1.keys(), c2.keys())

# Separate old/new
old_ids = sorted([i for i in all_ids if i in p1_old or i in p2 or i in p3],
                 key=lambda x: (0 if "astropy" in x else 1, x))
new_ids = sorted([i for i in all_ids if i in p1_new and i not in old_ids],
                 key=lambda x: x)

# Print comprehensive table
print("=" * 140)
print("FULL CROSS-PHASE COMPARISON: P1 → P2 → P3 → C1 → C2")
print("=" * 140)

print(f"\n{'Instance':<32} {'Set':<6} {'P1':<6} {'P2':<6} {'P3':<6} {'C1':<6} {'C2':<6} {'C1vP3':<8} {'C2vC1':<8} {'Diag':<6}")
print(f"{'-'*32} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")

# Initialize counters
counts = {"old": {"P1":0, "P2":0, "P3":0, "C1":0, "C2":0, "total":0},
          "new": {"P1":0, "C1":0, "C2":0, "total":0}}

flags_all = {"P1": {"D":0,"E":0,"A":0}, "P2": {"D":0,"E":0,"A":0},
             "P3": {"D":0,"E":0,"A":0}, "C1": {"D":0,"E":0,"A":0},
             "C2": {"D":0,"E":0,"A":0}}

for iid in old_ids:
    p1v = p1_old.get(iid, {}).get("verdict", "?")
    p2v = p2.get(iid, {}).get("verdict", "?")
    p3v = p3.get(iid, {}).get("verdict", "?")
    c1v = c1.get(iid, {}).get("verdict", "?")
    c2v = c2.get(iid, {}).get("verdict", "?")

    iset = "old36"
    p1f = p1_old.get(iid, {}).get("flags", [])
    p2f = p2.get(iid, {}).get("flags", [])
    p3f = p3.get(iid, {}).get("flags", [])
    c1f = c1.get(iid, {}).get("flags", [])
    c2f = c2.get(iid, {}).get("flags", [])
    for f in p1f: flags_all["P1"]["D" if f.startswith("D_") else "E" if f.startswith("E_") else "A"] += 1
    for f in p2f: flags_all["P2"]["D" if f.startswith("D_") else "E" if f.startswith("E_") else "A"] += 1
    for f in p3f: flags_all["P3"]["D" if f.startswith("D_") else "E" if f.startswith("E_") else "A"] += 1
    for f in c1f: flags_all["C1"]["D" if f.startswith("D_") else "E" if f.startswith("E_") else "A"] += 1
    for f in c2f: flags_all["C2"]["D" if f.startswith("D_") else "E" if f.startswith("E_") else "A"] += 1

    c1vp3 = ""
    if p3v != "?" and c1v != "?":
        if c1v == "PASS" and p3v == "REJECT": c1vp3 = "C1↑"
        elif c1v == "REJECT" and p3v == "PASS": c1vp3 = "P3↑"
        else: c1vp3 = "same"
    c2vc1 = ""
    if c1v != "?" and c2v != "?":
        if c2v == "PASS" and c1v == "REJECT": c2vc1 = "C2↑"
        elif c2v == "REJECT" and c1v == "PASS": c2vc1 = "C1↑"
        else: c2vc1 = "same"

    diag_info = ""
    if iid in c1_diag: diag_info += "1"
    if iid in c2_diag: diag_info += "2"

    counts["old"]["total"] += 1
    for ph, v in [("P1",p1v),("P2",p2v),("P3",p3v),("C1",c1v),("C2",c2v)]:
        if v == "PASS": counts["old"][ph] += 1

    print(f"{iid:<32} {iset:<6} {p1v:<6} {p2v:<6} {p3v:<6} {c1v:<6} {c2v:<6} {c1vp3:<8} {c2vc1:<8} {diag_info:<6}")

# Print new36
for iid in new_ids:
    p1v = p1_new.get(iid, {}).get("verdict", "?")
    c1v = c1.get(iid, {}).get("verdict", "?")
    c2v = c2.get(iid, {}).get("verdict", "?")

    iset = "new36"
    c1f = c1.get(iid, {}).get("flags", [])
    c2f = c2.get(iid, {}).get("flags", [])
    for f in c1f: flags_all["C1"]["D" if f.startswith("D_") else "E" if f.startswith("E_") else "A"] += 1
    for f in c2f: flags_all["C2"]["D" if f.startswith("D_") else "E" if f.startswith("E_") else "A"] += 1

    c2vc1 = ""
    if c1v != "?" and c2v != "?":
        if c2v == "PASS" and c1v == "REJECT": c2vc1 = "C2↑"
        elif c2v == "REJECT" and c1v == "PASS": c2vc1 = "C1↑"
        else: c2vc1 = "same"

    diag_info = ""
    if iid in c1_diag: diag_info += "1"
    if iid in c2_diag: diag_info += "2"

    counts["new"]["total"] += 1
    for ph, v in [("P1",p1v),("C1",c1v),("C2",c2v)]:
        if v == "PASS": counts["new"][ph] += 1

    print(f"{iid:<32} {iset:<6} {p1v:<6} {'?':<6} {'?':<6} {c1v:<6} {c2v:<6} {'':<8} {c2vc1:<8} {diag_info:<6}")

# ============================
# SUMMARY
# ============================
print(f"\n{'='*140}")
print("SUMMARY")
print(f"{'='*140}")

# Old 36
o = counts["old"]; n = counts["new"]
print(f"\n  Old 36 (indices 0-35, astropy+django):")
print(f"    P1: {o['P1']}/{o['total']} ({100*o['P1']/o['total']:.1f}%)")
print(f"    P2: {o['P2']}/{o['total']} ({100*o['P2']/o['total']:.1f}%)")
print(f"    P3: {o['P3']}/{o['total']} ({100*o['P3']/o['total']:.1f}%)")
print(f"    C1: {o['C1']}/{o['total']} ({100*o['C1']/o['total']:.1f}%)")
print(f"    C2: {o['C2']}/{o['total']} ({100*o['C2']/o['total']:.1f}%)")

print(f"\n  New 36 (indices 36-71, all django):")
print(f"    P1: {n['P1']}/{n['total']} ({100*n['P1']/n['total']:.1f}%)")
print(f"    C1: {n['C1']}/{n['total']} ({100*n['C1']/n['total']:.1f}%)")
print(f"    C2: {n['C2']}/{n['total']} ({100*n['C2']/n['total']:.1f}%)")

total_all = o["total"] + n["total"]
print(f"\n  Total (old36 + new36 = {total_all}):")
print(f"    C1: {o['C1']+n['C1']}/{total_all} ({100*(o['C1']+n['C1'])/total_all:.1f}%)")
print(f"    C2: {o['C2']+n['C2']}/{total_all} ({100*(o['C2']+n['C2'])/total_all:.1f}%)")
print(f"    Δ(C2 - C1): {(o['C2']+n['C2']) - (o['C1']+n['C1']):+d} PASS")

# Old36 only cross-phase effects
print(f"\n  Cross-phase Δ (old36 only):")
p1_p2 = o['P2'] - o['P1']
p2_p3 = o['P3'] - o['P2']
p1_p3 = o['P3'] - o['P1']
p3_c1 = o['C1'] - o['P3']
c1_c2 = o['C2'] - o['C1']
print(f"    P1→P2 (self-reflection): {p1_p2:+d} PASS ({100*p1_p2/o['total']:+.1f}pp)")
print(f"    P2→P3 (Mimo batch diag): {p2_p3:+d} PASS ({100*p2_p3/o['total']:+.1f}pp)")
print(f"    P1→P3 (total improvement): {p1_p3:+d} PASS ({100*p1_p3/o['total']:+.1f}pp)")
print(f"    P3→C1 (batch → real-time): {p3_c1:+d} PASS ({100*p3_c1/o['total']:+.1f}pp)")
print(f"    C1→C2 (immediate → SR buffer): {c1_c2:+d} PASS ({100*c1_c2/o['total']:+.1f}pp)")

# D-E-A flag comparison for old36
print(f"\n  D-E-A Flag distribution (old36):")
for ph in ["P1", "P2", "P3", "C1", "C2"]:
    f = flags_all[ph]
    d = f["D"]; e = f["E"]; a = f["A"]
    denom = max(d+a, 1)
    print(f"    {ph}: D={d} E={e} A={a} | E/(D+A)={e/denom:.2f}")

# Diag stats
print(f"\n  Diagnosis injection stats (all 72):")
print(f"    C1: {len(c1_diag)} instances received real-time external diagnosis")
print(f"    C2: {len(c2_diag)} instances received external diagnosis (after SR failed)")
print(f"    C2: {len(c2_sr)} instances triggered self-reflection")
print(f"    C2 SR without escalation to diag: {len(c2_sr) - len(c2_diag)}")

# C2 SR effectiveness
sr_pass_no_diag = 0
sr_reject_no_diag = 0
for iid in c2_sr:
    if iid not in c2_diag:
        if c2.get(iid, {}).get("verdict") == "PASS":
            sr_pass_no_diag += 1
        else:
            sr_reject_no_diag += 1
print(f"    C2 SR-only (no diag): PASS={sr_pass_no_diag} REJECT={sr_reject_no_diag}")

print(f"\n  Comparison → {BASE / 'turning_point_validation' / 'full_comparison.json'}")
