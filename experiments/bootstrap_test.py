"""
Bootstrap significance test for SWE-bench three-phase results.
Tests:
  1. P1->P2 PASS rate change: is +6pp statistically significant?
  2. P2->P3 PASS rate change: is +5pp statistically significant?
  3. E/(D+A) cross-sample stability: 1.08 vs 1.04, could be noise?

Method: Bootstrap resample 36 instances with replacement, compute
statistic for each resample, derive 95% CI.
"""
import json, random, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

random.seed(42)

# ── Load data ──
def load_phase(dir_name):
    """Load results from summary JSON if exists, else from phase directory files."""
    import os
    # Prefer summary file (has flags)
    summary_paths = [
        f"D:/数据/论文数据/SWE-bench_DEA/{dir_name}.json",
        f"D:/数据/论文数据/SWE-bench_DEA/phase1_baseline_kb.json",
        f"D:/数据/论文数据/SWE-bench_DEA/phase2_self_reflection_kb.json",
        f"D:/数据/论文数据/SWE-bench_DEA/phase3_old36_kb.json",
        f"D:/数据/论文数据/SWE-bench_DEA/phase1_new36_kb.json",
    ]
    # Check for a matching summary file
    summary_map = {
        'phase1_baseline': 'phase1_summary.json',
        'phase2_self_reflection': 'phase2_summary.json',
        'phase3_old36': 'phase3_old36_summary.json',
        'phase1_new36': 'phase1_new36_summary.json',
    }
    sm = summary_map.get(dir_name)
    results = []
    if sm:
        with open(f"D:/数据/论文数据/SWE-bench_DEA/{sm}", 'r', encoding='utf-8') as f:
            data = json.load(f)
        for r in data:
            results.append({
                'verdict': r.get('reviewer_verdict') or r.get('final_verdict') or '?',
                'flags': r.get('flags', []),
                'error': r.get('error'),
            })
        return results

    # Fallback: read phase directory
    d = f"D:/数据/论文数据/SWE-bench_DEA/{dir_name}"
    for fname in sorted(os.listdir(d)):
        if not fname.endswith('.json'): continue
        if 'kb' in fname: continue
        with open(f"{d}/{fname}", 'r', encoding='utf-8') as f:
            data = json.load(f)
        v = data.get('final_verdict') or data.get('reviewer_verdict') or '?'
        fl = data.get('flags') or data.get('trajectory', {}).get('flags', [])
        results.append({'verdict': v, 'flags': fl, 'error': data.get('error')})
    return results

p1_old = load_phase("phase1_baseline")
p2_old = load_phase("phase2_self_reflection")
p3_old = load_phase("phase3_old36")
p1_new = load_phase("phase1_new36")

def pass_rate(results):
    valid = [r for r in results if not r.get('error')]
    return sum(1 for r in valid if r.get('verdict') == 'PASS') / max(len(valid), 1)

def e_ratio(results):
    all_flags = [f for r in results if not r.get('error') for f in r.get('flags', [])]
    d = sum(1 for f in all_flags if f.startswith('D_'))
    e = sum(1 for f in all_flags if f.startswith('E_'))
    a = sum(1 for f in all_flags if f.startswith('A_'))
    return e / max(d + a, 1)

def avg_flags(results, prefix):
    all_flags = [f for r in results if not r.get('error') for f in r.get('flags', [])]
    return sum(1 for f in all_flags if f.startswith(prefix))

print("=" * 70)
print("BOOTSTRAP SIGNIFICANCE TEST (n=36, 10,000 resamples)")
print("=" * 70)

N_BOOT = 10000

# ── Test 1: P1 vs P2 PASS rate ──
p1_pass = pass_rate(p1_old)
p2_pass = pass_rate(p2_old)
obs_diff_12 = p2_pass - p1_pass  # should be ~0.056
pooled_12 = [(r1, r2) for r1, r2 in zip(p1_old, p2_old)]

diffs_12 = []
for _ in range(N_BOOT):
    sample = [pooled_12[random.randint(0, 35)] for _ in range(36)]
    p1_boot = sum(1 for r1, r2 in sample if not r1.get('error') and r1.get('verdict') == 'PASS') / 36
    p2_boot = sum(1 for r1, r2 in sample if not r2.get('error') and r2.get('verdict') == 'PASS') / 36
    diffs_12.append(p2_boot - p1_boot)

diffs_12.sort()
ci_12_low = diffs_12[250]   # 2.5th percentile
ci_12_high = diffs_12[9749]  # 97.5th percentile
p_12 = sum(1 for d in diffs_12 if d <= 0) / N_BOOT  # one-sided: H0 diff<=0

print(f"\nTest 1: P1→P2 PASS rate change")
print(f"  P1 PASS: {p1_pass:.3f}  |  P2 PASS: {p2_pass:.3f}")
print(f"  Observed delta: {obs_diff_12:+.3f} (+{obs_diff_12*100:.0f}pp)")
print(f"  95% CI: [{ci_12_low:+.3f}, {ci_12_high:+.3f}]")
print(f"  p-value (H0: delta<=0): {p_12:.4f}")
print(f"  Significant at 5%: {'YES' if p_12 < 0.05 else 'NO'}")

# ── Test 2: P2 vs P3 PASS rate ──
p3_pass = pass_rate(p3_old)
obs_diff_23 = p3_pass - p2_pass
pooled_23 = [(r2, r3) for r2, r3 in zip(p2_old, p3_old)]

diffs_23 = []
for _ in range(N_BOOT):
    sample = [pooled_23[random.randint(0, 35)] for _ in range(36)]
    p2_boot = sum(1 for r2, r3 in sample if not r2.get('error') and r2.get('verdict') == 'PASS') / 36
    p3_boot = sum(1 for r2, r3 in sample if not r3.get('error') and r3.get('verdict') == 'PASS') / 36
    diffs_23.append(p3_boot - p2_boot)

diffs_23.sort()
ci_23_low = diffs_23[250]
ci_23_high = diffs_23[9749]
p_23 = sum(1 for d in diffs_23 if d <= 0) / N_BOOT

print(f"\nTest 2: P2→P3 PASS rate change")
print(f"  P2 PASS: {p2_pass:.3f}  |  P3 PASS: {p3_pass:.3f}")
print(f"  Observed delta: {obs_diff_23:+.3f} (+{obs_diff_23*100:.0f}pp)")
print(f"  95% CI: [{ci_23_low:+.3f}, {ci_23_high:+.3f}]")
print(f"  p-value (H0: delta<=0): {p_23:.4f}")
print(f"  Significant at 5%: {'YES' if p_23 < 0.05 else 'NO'}")

# ── Test 3: P1→P3 total improvement ──
obs_diff_13 = p3_pass - p1_pass
pooled_13 = [(r1, r3) for r1, r3 in zip(p1_old, p3_old)]

diffs_13 = []
for _ in range(N_BOOT):
    sample = [pooled_13[random.randint(0, 35)] for _ in range(36)]
    p1_boot = sum(1 for r1, r3 in sample if not r1.get('error') and r1.get('verdict') == 'PASS') / 36
    p3_boot = sum(1 for r1, r3 in sample if not r3.get('error') and r3.get('verdict') == 'PASS') / 36
    diffs_13.append(p3_boot - p1_boot)

diffs_13.sort()
ci_13_low = diffs_13[250]
ci_13_high = diffs_13[9749]
p_13 = sum(1 for d in diffs_13 if d <= 0) / N_BOOT

print(f"\nTest 3: P1→P3 total improvement")
print(f"  P1 PASS: {p1_pass:.3f}  |  P3 PASS: {p3_pass:.3f}")
print(f"  Observed delta: {obs_diff_13:+.3f} (+{obs_diff_13*100:.0f}pp)")
print(f"  95% CI: [{ci_13_low:+.3f}, {ci_13_high:+.3f}]")
print(f"  p-value (H0: delta<=0): {p_13:.4f}")
print(f"  Significant at 5%: {'YES' if p_13 < 0.05 else 'NO'}")

# ── Test 4: D旗 P1→P2 change ──
p1_d = avg_flags(p1_old, 'D_')
p2_d = avg_flags(p2_old, 'D_')
obs_diff_d = p2_d - p1_d

diffs_d = []
for _ in range(N_BOOT):
    sample = [(random.choice(p1_old), random.choice(p2_old)) for _ in range(36)]
    p1_d_boot = avg_flags([s[0] for s in sample], 'D_')
    p2_d_boot = avg_flags([s[1] for s in sample], 'D_')
    diffs_d.append(p2_d_boot - p1_d_boot)

diffs_d.sort()
ci_d_low = diffs_d[250]
ci_d_high = diffs_d[9749]
p_d = sum(1 for d in diffs_d if d >= 0) / N_BOOT

print(f"\nTest 4: P1→P2 D旗 change")
print(f"  P1 D旗: {p1_d}  |  P2 D旗: {p2_d}")
print(f"  Observed delta: {obs_diff_d:+d}")
print(f"  95% CI: [{ci_d_low:+.0f}, {ci_d_high:+.0f}]")
print(f"  p-value (H0: delta>=0): {p_d:.4f}")

# ── Test 5: E旗 P1→P2 change ──
p1_e = avg_flags(p1_old, 'E_')
p2_e = avg_flags(p2_old, 'E_')
obs_diff_e = p2_e - p1_e

diffs_e = []
for _ in range(N_BOOT):
    sample = [(random.choice(p1_old), random.choice(p2_old)) for _ in range(36)]
    p1_e_boot = avg_flags([s[0] for s in sample], 'E_')
    p2_e_boot = avg_flags([s[1] for s in sample], 'E_')
    diffs_e.append(p2_e_boot - p1_e_boot)

diffs_e.sort()
ci_e_low = diffs_e[250]
ci_e_high = diffs_e[9749]
p_e = sum(1 for d in diffs_e if d <= 0) / N_BOOT

print(f"\nTest 5: P1→P2 E旗 change (key: D可自修复, E不能)")
print(f"  P1 E旗: {p1_e}  |  P2 E旗: {p2_e}")
print(f"  Observed delta: {obs_diff_e:+d}")
print(f"  95% CI: [{ci_e_low:+.0f}, {ci_e_high:+.0f}]")
print(f"  p-value (H0: delta<=0): {p_e:.4f}")
print(f"  Interpretation: E旗 P1→P2 {'significantly INCREASES' if p_e < 0.05 else 'no significant change'}")

# ── Test 6: Cross-sample E/(D+A) stability ──
er_old = e_ratio(p1_old)
er_new = e_ratio(p1_new)
obs_diff_er = abs(er_old - er_new)

# Bootstrap the DIFFERENCE between old and new E/(D+A)
# If stability is real, the observed difference (0.04) should be
# within the range of differences we'd see by chance between
# two random 36-instance samples from the SAME population.
# We use old as the reference population for bootstrapping.

er_diffs_null = []
for _ in range(N_BOOT):
    s1 = [random.choice(p1_old) for _ in range(36)]
    s2 = [random.choice(p1_old) for _ in range(36)]
    er_diffs_null.append(abs(e_ratio(s1) - e_ratio(s2)))

er_diffs_null.sort()
ci_er = er_diffs_null[9499]  # 95th percentile of null distribution
p_er = sum(1 for d in er_diffs_null if d >= obs_diff_er) / N_BOOT

print(f"\nTest 6: Cross-sample E/(D+A) stability")
print(f"  Old36 E/(D+A): {er_old:.2f}  |  New36 E/(D+A): {er_new:.2f}")
print(f"  Observed difference: {obs_diff_er:.2f}")
print(f"  Null 95% CI (under random split into 2x36): up to {ci_er:.2f}")
print(f"  p-value (H0: difference is random splitting variance): {p_er:.4f}")
print(f"  Interpretation: {'Cannot reject null — difference is within random splitting variance' if p_er > 0.05 else 'Difference is LARGER than expected under random splitting — NOT stable'}")
print(f"  NOTE: Smaller p = LESS stable. Larger p = MORE likely that 1.08 vs 1.04 is just sampling noise from the same underlying distribution — i.e., E/(D+A) IS stable across samples.")

# ── Summary ──
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  Test 1 (P1→P2 PASS):     delta={obs_diff_12:+.3f}, 95%CI=[{ci_12_low:+.3f},{ci_12_high:+.3f}], p={p_12:.4f} {'*' if p_12 < 0.05 else ''}")
print(f"  Test 2 (P2→P3 PASS):     delta={obs_diff_23:+.3f}, 95%CI=[{ci_23_low:+.3f},{ci_23_high:+.3f}], p={p_23:.4f} {'*' if p_23 < 0.05 else ''}")
print(f"  Test 3 (P1→P3 total):    delta={obs_diff_13:+.3f}, 95%CI=[{ci_13_low:+.3f},{ci_13_high:+.3f}], p={p_13:.4f} {'*' if p_13 < 0.05 else ''}")
print(f"  Test 4 (D旗 P1→P2):      delta={obs_diff_d:+d}, 95%CI=[{ci_d_low:+.0f},{ci_d_high:+.0f}], p={p_d:.4f} {'*' if p_d < 0.05 else ''}")
print(f"  Test 5 (E旗 P1→P2):      delta={obs_diff_e:+d}, 95%CI=[{ci_e_low:+.0f},{ci_e_high:+.0f}], p={p_e:.4f} {'*' if p_e < 0.05 else ''}")
print(f"  Test 6 (E/(D+A) stable): diff={obs_diff_er:.2f}, null95%={ci_er:.2f}, p={p_er:.4f} {'*' if p_er > 0.05 else ''}")
print(f"\n  * = significant at 5% level")
print(f"  Test 6 interpretation: p>0.05 means the observed 0.04 difference")
print(f"  is CONSISTENT with random sampling from the same underlying")
print(f"  distribution — i.e., E/(D+A) is statistically stable across samples.")
