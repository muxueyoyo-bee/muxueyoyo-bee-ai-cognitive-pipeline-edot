"""Extract D-E-A ratings for all 6 turning point trajectories."""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

out_dir = "D:/数据/论文数据/SWE-bench_DEA/turning_point_results"

rows = []
for fname in sorted(os.listdir(out_dir)):
    if not fname.endswith('.json') or 'kb' in fname or 'diagnosis_log' in fname or 'comparison' in fname:
        continue
    fpath = os.path.join(out_dir, fname)
    with open(fpath, 'r', encoding='utf-8') as fh:
        d = json.load(fh)

    t = d.get('trajectory', {})
    iid = d.get('instance_id', fname)
    verdict = d.get('reviewer_verdict', '?')
    revs = d.get('revision_count', 0)
    nsubs = d.get('num_subproblems', 0)
    flags = d.get('flags', [])
    diag = d.get('diagnosis_injected', False)
    diag_round = d.get('diagnosis_round', -1)
    diag_text = t.get('diagnosis_text', '')
    patch_chars = d.get('patch_chars', 0)

    print('=' * 70)
    print(f'INSTANCE: {iid}')
    print(f'VERDICT: {verdict} | REVISIONS: {revs} | SUBS: {nsubs} | PATCH: {patch_chars} chars')
    print(f'DIAGNOSIS: {"injected@A"+str(diag_round) if diag else "not triggered (A0 PASS or error)"}')
    print(f'HEURISTIC FLAGS: {flags}')

    # ── D-E-A coding ──
    d_issues = []
    e_issues = []
    a_issues = []
    severity = {'D': '无', 'E': '无', 'A': '无'}

    # D
    d_output = t.get('D_output', '')
    if nsubs == 0:
        d_issues.append('D_EMPTY — Locator未产出子问题')
        severity['D'] = '严重'
    elif nsubs == 1:
        d_issues.append('D_MONOLITH — 单一子问题，未拆解')
        severity['D'] = '轻度'
    elif nsubs >= 8:
        d_issues.append('D_OVERFRAGMENT — 过度拆解')
        severity['D'] = '中度'
    else:
        d_issues.append(f'拆解合理 ({nsubs}个子问题)')

    # E
    e_rounds = t.get('E_rounds', [])
    e_text = ' '.join(e_rounds) if e_rounds else ''
    if not e_text.strip() or e_text.strip().startswith('--- Revision'):
        e_issues.append('E_EMPTY — Fixer首轮无输出')
        severity['E'] = '严重'
    else:
        if '[ABANDONED]' not in e_text and nsubs >= 3:
            e_issues.append('E_NO_ABANDON — 未放弃低优先级子问题')
            severity['E'] = '轻度'
        if 'P0' not in e_text and nsubs >= 2:
            e_issues.append('E_FLAT_PRIORITY — 优先级扁平化')
            if severity['E'] == '无': severity['E'] = '轻度'
        if not any(kw in e_text.lower() for kw in ['because','reason','impact','blast','justif']):
            e_issues.append('E_NO_JUSTIFICATION — 无显式优先级论证')
            if severity['E'] == '无': severity['E'] = '轻度'

    # A
    a_response = t.get('A_response', '')
    final_patch = t.get('final_patch', '')
    if verdict == 'REJECT':
        a_issues.append(f'A_MAX_REVISIONS — {revs}轮修订→REJECT')
        severity['A'] = '严重'
    elif verdict == 'PASS' and patch_chars < 100:
        a_issues.append('A_FALSE_PASS — patch过短({patch_chars} chars)')
        severity['A'] = '中度'
    elif verdict == 'PASS':
        a_issues.append('审查通过')
    elif verdict == 'ERROR':
        a_issues.append('A_ERROR — 系统异常')
        severity['A'] = '严重'

    # Primary bottleneck
    sev_order = {'严重': 3, '中度': 2, '轻度': 1, '无': 0}
    bottleneck = '无'
    for key in ['E', 'A', 'D']:
        if sev_order[severity[key]] >= sev_order[severity.get(bottleneck, '无')]:
            if sev_order[severity[key]] > 0:
                bottleneck = key
    if all(sev_order[severity[k]] == 0 for k in ['D','E','A']):
        bottleneck = '无'

    print(f'\n### D 问题：{" | ".join(d_issues) if d_issues else "无"} | 严重性：{severity["D"]}')
    print(f'### E 问题：{" | ".join(e_issues) if e_issues else "无"} | 严重性：{severity["E"]}')
    print(f'### A 问题：{" | ".join(a_issues) if a_issues else "无"} | 严重性：{severity["A"]}')
    print(f'### 主要瓶颈：{bottleneck}')

    # ── Pro 诊断质量 ──
    if diag_text and 'Root Cause' in diag_text:
        print(f'\n### Pro诊断质量（实时注入@A{diag_round}）')
        has_dea = all(kw+':' in diag_text for kw in ['D', 'E', 'A'])
        guidance_count = sum(1 for line in diag_text.split('\n') if line.strip()[:2].rstrip('.').isdigit())
        print(f'  D-E-A框架: {"完整" if has_dea else "部分"} | 可操作指令: ~{guidance_count}条')
        # Was this systemic or local?
        if severity['E'] in ('中度','严重') or severity['A'] in ('中度','严重'):
            prob_areas = [k for k in ['D','E','A'] if severity[k] in ('中度','严重')]
            print(f'  问题层级: {"/".join(prob_areas)}环节缺陷 → 可能需要系统级prompt干预而非实例级指导')
    elif not diag:
        print(f'\n### 诊断: 未触发（A0即PASS，无需外部介入）')
    print()

    rows.append({
        'iid': iid, 'verdict': verdict, 'revs': revs, 'nsubs': nsubs,
        'D_sev': severity['D'], 'E_sev': severity['E'], 'A_sev': severity['A'],
        'bottleneck': bottleneck, 'diag': diag, 'flags_count': len(flags),
    })

# ── Summary ──
print('=' * 70)
print('D-E-A SUMMARY: Turning Point Experiment (n=6)')
print('=' * 70)
print(f"{'Instance':<30} {'D':<8} {'E':<8} {'A':<8} {'Bottleneck':<12} {'Verdict':<8} {'Diag?':<6}")
print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*8} {'-'*6}")
for r in rows:
    print(f"{r['iid']:<30} {r['D_sev']:<8} {r['E_sev']:<8} {r['A_sev']:<8} {r['bottleneck']:<12} {r['verdict']:<8} {'Y' if r['diag'] else 'N':<6}")

# Aggregate
d_severe = sum(1 for r in rows if r['D_sev'] in ('中度','严重'))
e_severe = sum(1 for r in rows if r['E_sev'] in ('中度','严重'))
a_severe = sum(1 for r in rows if r['A_sev'] in ('中度','严重'))
bottlenecks = [r['bottleneck'] for r in rows if r['bottleneck'] != '无']
bn_counts = {b: bottlenecks.count(b) for b in ['D','E','A']}

print(f'\n--- Aggregate ---')
print(f'  D严重: {d_severe}/6 | E严重: {e_severe}/6 | A严重: {a_severe}/6')
print(f'  瓶颈分布: D={bn_counts["D"]} E={bn_counts["E"]} A={bn_counts["A"]} 无={6-sum(bn_counts.values())}')
print(f'  PASS: {sum(1 for r in rows if r["verdict"]=="PASS")}/6')
print(f'  Diag triggered: {sum(1 for r in rows if r["diag"])}/6')
