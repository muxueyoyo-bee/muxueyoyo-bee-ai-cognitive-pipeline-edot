# Memory Injection Prompt Template v2.0

## Purpose

Injected into every agent call context to carry forward methodology improvements from previous batches.

## Injection Format

```
## Knowledge Base (accumulated from previous instances)
- [System] methodology_rule_MR001: P0 标签必须附带至少一句论证（scope: E环节）
- [System] methodology_rule_MR002: 子问题 >3 时必须标记至少一个 [ABANDONED]（scope: A环节）
- [System] methodology_rule_MR004: 拆解时必须显式枚举独立关注点（scope: D环节）
- [Fixer] instance_15_summary: {"num_subproblems": 5, "verdict": "PASS", "flags": ["D_MONOLITH", "E_NO_JUSTIFICATION"]}
- [Reviewer] instance_18_pattern: cross-instance priority inconsistency on config-type bugs
```

## Rules

1. Maximum 5 entries per injection (controlled by `memory_injection.yaml`)
2. Entries sorted by recency (most recent first)
3. Methodology rules (MR00X) take precedence over instance summaries
4. Each entry max 400 characters (truncated if longer)
5. KB content is READ-ONLY to execution agents — they use it for context but do not modify it

## Configuration

See `configs/memory_injection.yaml` for:
- Injection trigger conditions
- Entry filtering rules (min confidence, min cross-instance count)
- Human review requirements
- Pruning policy
