"""Count all comment data across directories."""
import json, os, glob

def count_summary_jsons(root):
    """Count comments from all summary.json files under root."""
    total = 0
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        for f in filenames:
            if 'summary' in f.lower() and f.endswith('.json'):
                fp = os.path.join(dirpath, f)
                for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb18030']:
                    try:
                        with open(fp, encoding=enc) as fh:
                            data = json.load(fh)
                        break
                    except:
                        continue
                tc = data.get('total_comments', data.get('total', 0))
                bv = data.get('bv_id', data.get('bv', os.path.basename(dirpath)))
                total += int(tc) if tc else 0
                results.append((bv, int(tc) if tc else 0))
    return results, total

def count_json_outputs(root):
    """Count comments from all JSON output files."""
    total = 0
    results = []
    for f in glob.glob(os.path.join(root, "*.json")):
        bn = os.path.basename(f)
        if any(x in bn for x in ['checkpoint', 'kb', 'summary', 'diag_log']):
            continue
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb18030']:
            try:
                with open(f, encoding=enc) as fh:
                    data = json.load(fh)
                break
            except:
                continue
        bs = data.get('basic_stats', {})
        tc = bs.get('total', 0)
        total += int(tc) if tc else 0
        results.append((bn, int(tc) if tc else 0))
    return results, total

def count_jsonl(root):
    """Count lines in jsonl files."""
    total = 0
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        for f in filenames:
            if f.endswith('.jsonl'):
                fp = os.path.join(dirpath, f)
                with open(fp, encoding='utf-8') as fh:
                    n = sum(1 for _ in fh)
                total += n
                results.append((os.path.basename(fp), n))
    return results, total

# 1. 崩铁/数据收集 (migrated)
print("=" * 60)
print("1. 崩铁/数据收集 BV子目录 (summary.json)")
r1, t1 = count_summary_jsons("D:/分析报告/崩铁/数据收集")
for bv, n in sorted(r1):
    print(f"  {bv}: {n}条")
print(f"  小计: {t1}条\n")

# 2. 崩铁/JSON输出
print("=" * 60)
print("2. 崩铁/JSON输出")
r2, t2 = count_json_outputs("D:/分析报告/崩铁/JSON输出")
for bv, n in sorted(r2):
    print(f"  {bv}: {n}条")
print(f"  小计: {t2}条 (含v1/v2重复)\n")

# 3. 数据收集 (original, pre-migration)
print("=" * 60)
print("3. 数据收集 (原始目录, 迁移前)")
r3, t3 = count_summary_jsons("D:/分析报告/数据收集")
for bv, n in sorted(r3):
    print(f"  {bv}: {n}条")
print(f"  小计: {t3}条\n")

# 4. 崩铁/bili jsonl
print("=" * 60)
print("4. 崩铁/数据收集/bili (jsonl日采)")
r4, t4 = count_jsonl("D:/分析报告/崩铁/数据收集/bili")
for fn, n in r4:
    print(f"  {fn}: {n}行")
print(f"  小计: {t4}行\n")

# 5. 数据收集/bili jsonl
print("=" * 60)
print("5. 数据收集/bili (jsonl)")
r5, t5 = count_jsonl("D:/分析报告/数据收集/bili")
for fn, n in r5:
    print(f"  {fn}: {n}行")
print(f"  小计: {t5}行\n")

# 6. 原神
print("=" * 60)
print("6. 原神/JSON输出")
r6, t6 = count_json_outputs("D:/分析报告/原神/JSON输出")
for bv, n in sorted(r6):
    print(f"  {bv}: {n}条")
print(f"  小计: {t6}条\n")

# Summary
print("=" * 60)
print("汇总")
print(f"  崩铁 BV子目录 summary: {t1}条")
print(f"  崩铁 JSON输出(含重复): {t2}条")
print(f"  原数据目录 summary: {t3}条")
print(f"  崩铁 bili jsonl: {t4}行")
print(f"  原数据目录 bili jsonl: {t5}行")
print(f"  原神 JSON: {t6}条")
