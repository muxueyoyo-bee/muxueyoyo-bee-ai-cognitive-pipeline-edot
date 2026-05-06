"""
Separation Experiment: Isolate diagnosis TIMING from diagnosis MODEL.

Blind spot (a): P3 changed both timing (batch) AND model (Mimo) simultaneously.
This experiment fixes V4Pro as the sole diagnoser and varies only timing.

Design:
  - 72 instances (old36 + new36), Phase 2 prompts, V4 Flash execution
  - Real-time condition: V4Pro diagnoses at first A0 REJECT → instance-level fix
  - Batch condition: after all 72 complete, V4Pro reviews ALL trajectories → D-E-A coding
  - 72 paired diagnosis records → 144 total

2x2 factorial completion:
                Batch              Real-time
  Mimo      ✅ P3 old36          (no API)
  V4Pro     ✅ NEW (n=72)       ✅ NEW (n=72)

Output: D:\数据\论文数据\SWE-bench_DEA\separation_experiment\
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
from agent_system import (
    DeepSeekLLM, KnowledgeBase, AgentPrompts, PROMPTS_V1,
    DEAPipeline,
)

# ── Output directories ─────────────────────────────────────────
BASE_OUT = Path("D:/数据/论文数据/SWE-bench_DEA/separation_experiment")
REALTIME_DIR = BASE_OUT / "realtime_v4pro"
BATCH_DIR = BASE_OUT / "batch_v4pro"
for d in [BASE_OUT, REALTIME_DIR, BATCH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Instance loading ───────────────────────────────────────────

def load_all_72_instances() -> list[dict]:
    from datasets import load_from_disk
    ds = load_from_disk("D:/数据/论文数据/SWE-bench_Verified")
    all_instances = list(ds)
    old_36 = all_instances[:36]
    new_36 = all_instances[36:72]
    for inst in old_36:
        inst["_set"] = "old36"
    for inst in new_36:
        inst["_set"] = "new36"
    instances = old_36 + new_36
    print(f"Loaded {len(old_36)} old + {len(new_36)} new = {len(instances)} total instances.")
    return instances


# ── Load Phase 2 prompts ───────────────────────────────────────

def load_phase2_prompts() -> AgentPrompts:
    prompts_file = Path("D:/数据/论文数据/SWE-bench_DEA/prompts_v2_for_mimo_review.json")
    if prompts_file.exists():
        with open(prompts_file, "r", encoding="utf-8") as f:
            p2_dict = json.load(f)
        return AgentPrompts(**p2_dict)
    else:
        print("WARNING: Phase 2 prompts not found, using V1 prompts.")
        return PROMPTS_V1


# ── External Diagnoser (V4Pro) ─────────────────────────────────

class ExternalDiagnoser:
    """DeepSeek V4 Pro diagnoses trajectories."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        self.model = "deepseek-v4-pro"

    def diagnose_realtime(self, instance_id: str, problem: str, d_output: str,
                          e_output: str, a_output: str) -> str:
        """Real-time: diagnose a single failing instance."""
        system_prompt = """You are an **External Diagnoser** — a stronger model called in to rescue a failing bug-fix attempt.

A multi-agent system (Locator→Fixer→Reviewer) has just produced a REJECT on a GitHub issue.
Your job: diagnose WHAT went wrong (D/E/A) and give the Fixer CONCRETE guidance.

Analyze using the D-E-A framework:
- **D (Decoupling)**: Were sub-problems properly decomposed? Wrong granularity?
- **E (Evaluation)**: Was priority justified with explicit criteria? Abandonment issues?
- **A (Aggregation)**: Did the patch address the issue? Is it complete?

Output format:
```
## D-E-A Diagnosis
D: <brief>
E: <brief>
A: <brief>

## Root Cause
<single most critical failure>

## Fixer Guidance
<3-5 concrete, specific instructions for the Fixer>
```
"""
        user_prompt = f"""## Instance: {instance_id}

## Original Issue
{problem[:3000]}

## Locator Output (D — Decoupling)
{d_output[:2000]}

## Fixer Output (E — Evaluation + Patch)
{e_output[:3000]}

## Reviewer Output (A — REJECTED)
{a_output[:2000]}

Diagnose why this failed and tell the Fixer exactly what to do differently.
"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
            stream=False,
        )
        return resp.choices[0].message.content

    def diagnose_batch(self, trajectories: list[dict]) -> str:
        """Batch: review ALL trajectories at once for systemic D-E-A patterns."""
        system_prompt = """You are an **External Diagnoser** conducting a BATCH review of a multi-agent bug-fixing system.

You will be given ALL trajectories from 72 instances. Your job is NOT instance-level debugging.
Instead, identify SYSTEMIC D-E-A patterns across the entire batch.

## D-E-A Framework
- **D (Decoupling)**: Are issues properly decomposed? Is there systematic monolithization?
- **E (Evaluation)**: Are priorities justified with explicit criteria? Is there systematic flat-priority?
- **A (Aggregation)**: Are patches addressing issues? Is the Reviewer catching problems?

## Output Format
```
## Batch Overview
Total instances: N
Overall PASS rate: X/N (Y%)
Total D flags: N_D | E flags: N_E | A flags: N_A
E/(D+A) ratio: Z.ZZ

## Systemic D Patterns
<patterns visible across multiple instances>

## Systemic E Patterns
<patterns visible across multiple instances>

## Systemic A Patterns
<patterns visible across multiple instances>

## Per-Instance D-E-A Coding
For EACH instance, provide:
### <instance_id>
D: <none|mild|moderate|severe> — <brief>
E: <none|mild|moderate|severe> — <brief>
A: <none|mild|moderate|severe> — <brief>
Primary bottleneck: [D|E|A]
Key observation: <one sentence>

## Cross-Instance Comparison
<how patterns differ between old36 (astropy+django) and new36 (django only)>

## Systemic Improvement Recommendations
### Locator Prompt Changes
<specific modifications>

### Fixer Prompt Changes
<specific modifications>

### Reviewer Prompt Changes
<specific modifications>
```
"""
        # Build compact trajectory summaries for batch review
        summaries = []
        for t in trajectories:
            inst_id = t.get("instance_id", "?")
            inst_set = t.get("_set", "?")
            repo = t.get("repo", "?")
            problem = t.get("problem_statement", "")[:600]
            verdict = t.get("reviewer_verdict", "?")
            flags = t.get("flags", [])
            n_subs = t.get("num_subproblems", 0)
            n_revs = t.get("revision_count", 0)
            diag_injected = t.get("diagnosis_injected", False)
            diag_preview = t.get("diagnosis_preview", "")[:300]

            traj = t.get("trajectory", {})
            d_out = traj.get("D_output", "")[:500]
            e_rounds = traj.get("E_rounds", [])
            e_last = e_rounds[-1][:500] if e_rounds else ""
            a_out = traj.get("A_response", "")[:300]

            summaries.append(f"""
### {inst_id} [{inst_set}] {repo}
Verdict: {verdict} | Subproblems: {n_subs} | Revisions: {n_revs}
Flags: {flags}
Real-time diag injected: {diag_injected}

D_output preview:
{d_out}

E_output preview (last round):
{e_last}

A_output preview:
{a_out}
---
""")

        # Truncate if too long (V4Pro context ~128k)
        joined = "\n".join(summaries)
        if len(joined) > 80000:
            joined = joined[:80000] + "\n\n[... truncated for context limit ...]"

        user_prompt = f"""## Batch Trajectory Review (72 instances)

Below are summaries of all 72 instance trajectories. Perform systemic D-E-A analysis.

{joined}

Analyze the full batch for systemic patterns and provide per-instance D-E-A coding.
"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=8192,
            stream=False,
        )
        return resp.choices[0].message.content


# ── Real-time Pipeline ─────────────────────────────────────────

class RealtimePipeline(DEAPipeline):
    """V4Pro real-time diagnosis at first A0 REJECT."""

    def __init__(self, llm: DeepSeekLLM, prompts: AgentPrompts, kb: KnowledgeBase,
                 diagnoser: ExternalDiagnoser, max_revisions: int = 2):
        super().__init__(llm, prompts, kb, max_revisions)
        self.diagnoser = diagnoser
        self.diag_log = []

    def run_one(self, instance: dict) -> dict:
        inst_id = instance["instance_id"]
        repo = instance["repo"]
        problem = instance["problem_statement"]
        instance_set = instance.get("_set", "?")

        kb_ctx = self.kb.get_kb_context(max_entries=5)
        kb_summary = self.kb.get_pattern_summary()
        print(f"  [{inst_id}] ({kb_summary}) D...", end="", flush=True)

        d_user = f"{kb_ctx}\n\n## Current Issue\n{problem}"
        d_response = self.llm.chat(self.prompts.locator, d_user)
        subproblems = self._parse_subproblems(d_response)
        self.kb.add("Locator", inst_id, "subproblems", json.dumps(subproblems, ensure_ascii=False))
        print(f" {len(subproblems)} subs → E...", end="", flush=True)

        e_user = (f"{kb_ctx}\n\n## Current Issue\n{problem}\n\n"
                  f"## Locator Output\n{d_response}")
        e_responses = [self.llm.chat(self.prompts.fixer, e_user)]
        current_patch = self._extract_patch(e_responses[0])

        revision = 0
        final_verdict = "REJECT"
        a_response = ""
        diag_injected = False
        diag_text = ""
        diag_round = -1

        while revision <= self.max_revisions:
            print(f" A{revision}...", end="", flush=True)
            a_user = (f"{kb_ctx}\n\n## Original Issue\n{problem}\n\n"
                      f"## Fixer Output\n{e_responses[-1]}\n")
            a_response = self.llm.chat(self.prompts.reviewer, a_user)
            verdict = self._parse_verdict(a_response)

            if verdict == "PASS":
                final_verdict = "PASS"
                break
            elif revision < self.max_revisions:
                if not diag_injected:
                    print(f" [V4Pro-RT]...", end="", flush=True)
                    try:
                        diag_text = self.diagnoser.diagnose_realtime(
                            inst_id, problem, d_response,
                            e_responses[-1], a_response
                        )
                        diag_injected = True
                        diag_round = revision
                        self.diag_log.append({
                            "instance_id": inst_id, "condition": "realtime_v4pro",
                            "round": revision, "diagnosis": diag_text[:1500],
                            "timestamp": datetime.now().isoformat(),
                        })
                        print(f" OK", end="", flush=True)
                    except Exception as e:
                        print(f" FAIL({e})", end="", flush=True)
                        diag_text = f"[Diagnosis failed: {e}]"

                if diag_text:
                    f_user = (f"{kb_ctx}\n\n## Original Issue\n{problem}\n\n"
                              f"## Reviewer REJECTED your patch\n{a_response}\n\n"
                              f"## [V4Pro REAL-TIME DIAGNOSIS]\n{diag_text}\n\n"
                              f"## Your previous output\n{e_responses[-1]}\n\n"
                              f"Revise your patch addressing ALL reviewer concerns "
                              f"AND the diagnosis guidance.")
                else:
                    f_user = (f"{kb_ctx}\n\n## Original Issue\n{problem}\n\n"
                              f"## Reviewer REJECTED your patch\n{a_response}\n\n"
                              f"## Your previous output\n{e_responses[-1]}\n\n"
                              f"Revise your patch addressing ALL reviewer concerns.")

                f_response = self.llm.chat(self.prompts.fixer, f_user, temperature=0.4)
                current_patch = self._extract_patch(f_response)
                e_responses.append(f"--- Revision {revision+1} ---\n{f_response}")
                revision += 1
            else:
                break

        print(f" {final_verdict}", flush=True)
        flags = self._flag(subproblems, e_responses, final_verdict, current_patch)
        self.kb.add_instance_summary(inst_id, len(subproblems), final_verdict,
                                      revision, flags, d_response)
        if current_patch:
            self.kb.add("Fixer", inst_id, "patch_preview", current_patch[:300])
        if final_verdict == "REJECT":
            self.kb.add("Reviewer", inst_id, "rejection_reason",
                        a_response[:300] if a_response else "no response")

        return {
            "instance_id": inst_id, "repo": repo, "_set": instance_set,
            "num_subproblems": len(subproblems), "subproblems": subproblems,
            "reviewer_verdict": final_verdict, "revision_count": revision,
            "patch_chars": len(current_patch), "flags": flags,
            "diagnosis_injected": diag_injected, "diagnosis_round": diag_round,
            "diagnosis_preview": diag_text[:500] if diag_text else "",
            "condition": "realtime_v4pro",
            "problem_statement": problem,
            "trajectory": {
                "instance_id": inst_id, "repo": repo,
                "problem_statement": problem,
                "D_output": d_response, "E_rounds": e_responses,
                "A_response": a_response, "final_verdict": final_verdict,
                "final_patch": current_patch[:2000],
                "diagnosis_injected": diag_injected,
                "diagnosis_text": diag_text[:2000],
            }
        }


# ── Batch Diagnoser (post-hoc on all trajectories) ─────────────

class BatchDiagnosisRunner:
    """After all 72 instances complete, run V4Pro batch review."""

    def __init__(self, diagnoser: ExternalDiagnoser):
        self.diagnoser = diagnoser

    def run(self, results: list[dict]) -> dict:
        print("\n" + "=" * 70)
        print("BATCH DIAGNOSIS: V4Pro reviewing all 72 trajectories at once...")
        print("=" * 70)

        valid = [r for r in results if not r.get("error")]
        print(f"  Valid trajectories: {len(valid)}/{len(results)}")

        batch_diagnosis = self.diagnoser.diagnose_batch(valid)

        # Parse per-instance D-E-A coding from batch output
        per_instance_coding = self._parse_per_instance_coding(batch_diagnosis, valid)

        output = {
            "condition": "batch_v4pro",
            "n_trajectories_reviewed": len(valid),
            "batch_diagnosis_full": batch_diagnosis,
            "per_instance_coding": per_instance_coding,
            "timestamp": datetime.now().isoformat(),
        }

        # Save
        batch_file = BATCH_DIR / "batch_diagnosis.json"
        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        batch_text_file = BATCH_DIR / "batch_diagnosis.md"
        with open(batch_text_file, "w", encoding="utf-8") as f:
            f.write(f"# V4Pro Batch Diagnosis\n\n")
            f.write(f"Reviewed: {len(valid)} trajectories\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            f.write(batch_diagnosis)

        print(f"  Batch diagnosis → {batch_file}")
        print(f"  Batch diagnosis (md) → {batch_text_file}")
        return output

    def _parse_per_instance_coding(self, batch_text: str, results: list[dict]) -> list[dict]:
        """Extract per-instance D-E-A coding from batch diagnosis text."""
        import re
        codings = []
        id_lookup = {r["instance_id"]: r for r in results}

        # Match patterns like: ### django__django-11149
        # followed by D: / E: / A: / Primary bottleneck: lines
        instance_blocks = re.split(r'\n### (?=\w+__\w+-\d+)', batch_text)

        for block in instance_blocks:
            if not block.strip():
                continue
            lines = block.strip().split('\n')
            inst_id = lines[0].strip() if lines else ""

            if inst_id not in id_lookup:
                # Try to extract instance_id from block start
                m = re.match(r'(\w+__\w+-\d+)', inst_id)
                if m:
                    inst_id = m.group(1)
                else:
                    continue

            d_severity = ""
            e_severity = ""
            a_severity = ""
            bottleneck = ""
            observation = ""

            for line in lines[1:]:
                if line.startswith("D:") or line.startswith("D —"):
                    d_severity = line.strip()
                elif line.startswith("E:") or line.startswith("E —"):
                    e_severity = line.strip()
                elif line.startswith("A:") or line.startswith("A —"):
                    a_severity = line.strip()
                elif "bottleneck" in line.lower() or "Primary bottleneck" in line:
                    bottleneck = line.strip()
                elif "observation" in line.lower() or "Key observation" in line:
                    observation = line.strip()

            codings.append({
                "instance_id": inst_id,
                "D_coding": d_severity,
                "E_coding": e_severity,
                "A_coding": a_severity,
                "primary_bottleneck": bottleneck,
                "key_observation": observation,
            })

        print(f"  Parsed {len(codings)} per-instance D-E-A codings from batch diagnosis.")
        return codings


# ── Cross-condition comparison ─────────────────────────────────

def build_comparison(realtime_results: list[dict], batch_output: dict):
    """Compare real-time vs batch V4Pro diagnosis."""
    print("\n" + "=" * 70)
    print("SEPARATION EXPERIMENT: Real-time vs Batch V4Pro Diagnosis")
    print("=" * 70)

    valid = [r for r in realtime_results if not r.get("error")]
    passes = sum(1 for r in valid if r.get("reviewer_verdict") == "PASS")
    total = len(valid)

    all_flags = [f for r in valid for f in r.get("flags", [])]
    d_count = sum(1 for f in all_flags if f.startswith("D_"))
    e_count = sum(1 for f in all_flags if f.startswith("E_"))
    a_count = sum(1 for f in all_flags if f.startswith("A_"))

    diag_count = sum(1 for r in valid if r.get("diagnosis_injected"))

    # Batch per-instance coding stats
    batch_codings = batch_output.get("per_instance_coding", [])
    batch_d_severe = sum(1 for c in batch_codings if "severe" in c.get("D_coding", "").lower())
    batch_e_severe = sum(1 for c in batch_codings if "severe" in c.get("E_coding", "").lower())
    batch_a_severe = sum(1 for c in batch_codings if "severe" in c.get("A_coding", "").lower())
    batch_e_bottleneck = sum(1 for c in batch_codings if "E" in c.get("primary_bottleneck", ""))

    # By set
    old36 = [r for r in valid if r.get("_set") == "old36"]
    new36 = [r for r in valid if r.get("_set") == "new36"]
    old_pass = sum(1 for r in old36 if r.get("reviewer_verdict") == "PASS")
    new_pass = sum(1 for r in new36 if r.get("reviewer_verdict") == "PASS")

    print(f"""
┌─────────────────────────────────────────────────────────────┐
│ REAL-TIME V4Pro (n={total})                                  │
├─────────────────────────────────────────────────────────────┤
│ PASS rate:      {passes}/{total} ({100*passes/max(total,1):.1f}%)                      │
│ Flag dist:      D={d_count} | E={e_count} | A={a_count}                    │
│ E/(D+A):        {e_count/max(d_count+a_count,1):.2f}                                    │
│ Diag injected:  {diag_count}/{total} instances                         │
│ Old36 PASS:     {old_pass}/{len(old36)} ({100*old_pass/max(len(old36),1):.1f}%)                    │
│ New36 PASS:     {new_pass}/{len(new36)} ({100*new_pass/max(len(new36),1):.1f}%)                    │
├─────────────────────────────────────────────────────────────┤
│ BATCH V4Pro (n={len(batch_codings)})                         │
├─────────────────────────────────────────────────────────────┤
│ D severe:       {batch_d_severe}                              │
│ E severe:       {batch_e_severe}                              │
│ A severe:       {batch_a_severe}                              │
│ E bottleneck:   {batch_e_bottleneck}/{len(batch_codings)}                              │
└─────────────────────────────────────────────────────────────┘
""")

    # Save comparison
    comparison = {
        "experiment": "separation_timing_vs_model",
        "diagnosis_model": "DeepSeek V4 Pro",
        "execution_model": "DeepSeek V4 Flash",
        "prompts": "Phase 2 (v2.0 self-reflection)",
        "realtime": {
            "n": total,
            "pass_rate": f"{passes}/{total} ({100*passes/max(total,1):.1f}%)",
            "flag_distribution": f"D={d_count} | E={e_count} | A={a_count}",
            "E_ratio": round(e_count/max(d_count+a_count,1), 2),
            "diag_injections": diag_count,
            "old36_pass": f"{old_pass}/{len(old36)}",
            "new36_pass": f"{new_pass}/{len(new36)}",
        },
        "batch": {
            "n_trajectories": len(batch_codings),
            "d_severe": batch_d_severe,
            "e_severe": batch_e_severe,
            "a_severe": batch_a_severe,
            "e_primary_bottleneck": f"{batch_e_bottleneck}/{len(batch_codings)}",
        },
        "comparison_notes": [
            "Same diagnoser (V4Pro), same prompts (Phase 2), same instances (72).",
            "Only difference: real-time (instance-level at A0 REJECT) vs batch (all trajectories at once).",
            "Compare with P3 Mimo batch to isolate model effect (Mimo vs V4Pro, same batch timing).",
            "Compare real-time vs batch within this experiment to isolate timing effect (same V4Pro model).",
        ],
        "timestamp": datetime.now().isoformat(),
    }

    comp_file = BASE_OUT / "separation_comparison.json"
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"  Comparison → {comp_file}")
    return comparison


# ── Main ───────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Separation Experiment: Timing vs Model")
    parser.add_argument("--realtime-only", action="store_true",
                       help="Only run real-time V4Pro diagnosis phase")
    parser.add_argument("--batch-only", action="store_true",
                       help="Only run batch V4Pro diagnosis (requires realtime results)")
    parser.add_argument("--compare-only", action="store_true",
                       help="Only build comparison from existing results")
    parser.add_argument("--start", type=int, default=0,
                       help="Start from instance index (for resume)")
    parser.add_argument("--limit", type=int, default=72,
                       help="Max instances to run")
    args = parser.parse_args()

    instances = load_all_72_instances()
    instances = instances[args.start:args.start + args.limit]

    if args.compare_only:
        rt_file = REALTIME_DIR / "realtime_summary.json"
        batch_file = BATCH_DIR / "batch_diagnosis.json"
        if rt_file.exists() and batch_file.exists():
            with open(rt_file, "r", encoding="utf-8") as f:
                rt_results = json.load(f)
            with open(batch_file, "r", encoding="utf-8") as f:
                batch_output = json.load(f)
            build_comparison(rt_results, batch_output)
        else:
            print("Missing result files. Run experiment first.")
        return

    if args.batch_only:
        # Load realtime results and run batch only
        rt_file = REALTIME_DIR / "realtime_summary.json"
        if not rt_file.exists():
            print("ERROR: realtime results not found. Run --realtime-only first.")
            return
        with open(rt_file, "r", encoding="utf-8") as f:
            rt_results = json.load(f)
        diagnoser = ExternalDiagnoser()
        batch_runner = BatchDiagnosisRunner(diagnoser)
        batch_output = batch_runner.run(rt_results)
        build_comparison(rt_results, batch_output)
        return

    # Load Phase 2 prompts
    prompts = load_phase2_prompts()
    print(f"Using prompts: {prompts.version}")

    exec_llm = DeepSeekLLM()
    diagnoser = ExternalDiagnoser()
    kb = KnowledgeBase(persistent_file=str(REALTIME_DIR / "realtime_kb.json"))

    print(f"\n{'='*70}")
    print("PHASE 1: REAL-TIME V4Pro DIAGNOSIS (72 instances)")
    print(f"  Execution: DeepSeek V4 Flash")
    print(f"  Diagnoser: DeepSeek V4 Pro (at A0 REJECT)")
    print(f"  Prompts: {prompts.version}")
    print(f"  Output: {REALTIME_DIR}")
    print(f"{'='*70}\n")

    pipeline = RealtimePipeline(exec_llm, prompts, kb, diagnoser)

    realtime_results = []
    for i, inst in enumerate(instances):
        inst_id = inst["instance_id"]
        print(f"[RT] {i+1}/{len(instances)}", end="")
        try:
            result = pipeline.run_one(inst)
            result["global_index"] = i + args.start
            realtime_results.append(result)

            traj_file = REALTIME_DIR / f"{inst_id.replace('/', '_')}.json"
            with open(traj_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            if (i + 1) % 5 == 0:
                kb.save(str(REALTIME_DIR / "realtime_kb.json"))
                with open(REALTIME_DIR / "realtime_checkpoint.json", "w", encoding="utf-8") as f:
                    json.dump({"last_idx": i, "n_results": len(realtime_results)}, f)
                print(f"  [checkpoint @ {i+1}]")
        except Exception as e:
            print(f"  !! Failed: {e}")
            import traceback
            traceback.print_exc()
            realtime_results.append({
                "instance_id": inst_id, "condition": "realtime_v4pro",
                "global_index": i + args.start,
                "error": str(e), "flags": ["EXEC_ERROR"], "reviewer_verdict": "ERROR"
            })

    # Save realtime results
    kb.save(str(REALTIME_DIR / "realtime_kb.json"))
    with open(REALTIME_DIR / "realtime_summary.json", "w", encoding="utf-8") as f:
        json.dump(realtime_results, f, ensure_ascii=False, indent=2)
    with open(REALTIME_DIR / "realtime_diag_log.json", "w", encoding="utf-8") as f:
        json.dump(pipeline.diag_log, f, ensure_ascii=False, indent=2)

    print(f"\n  Real-time phase complete: {len(realtime_results)} instances")
    print(f"  Summary → {REALTIME_DIR / 'realtime_summary.json'}")
    print(f"  Diag log → {REALTIME_DIR / 'realtime_diag_log.json'}")

    # ── BATCH PHASE ──
    print(f"\n{'='*70}")
    print("PHASE 2: BATCH V4Pro DIAGNOSIS (all 72 trajectories)")
    print(f"  Diagnoser: DeepSeek V4 Pro (reviews all at once)")
    print(f"  Output: {BATCH_DIR}")
    print(f"{'='*70}")

    batch_runner = BatchDiagnosisRunner(diagnoser)
    batch_output = batch_runner.run(realtime_results)

    # ── COMPARISON ──
    build_comparison(realtime_results, batch_output)

    print(f"\n{'='*70}")
    print("SEPARATION EXPERIMENT COMPLETE")
    print(f"  Real-time results: {REALTIME_DIR}")
    print(f"  Batch diagnosis:   {BATCH_DIR}")
    print(f"  Comparison:        {BASE_OUT / 'separation_comparison.json'}")
    print(f"{'='*70}")
    print(f"\n  2x2 factorial now complete:")
    print(f"                  Batch              Real-time")
    print(f"    Mimo      ✅ P3 old36          (no API)")
    print(f"    V4Pro     ✅ NEW (n={len(valid := [r for r in realtime_results if not r.get('error')])})           ✅ NEW (n={len(valid)})")
    print(f"\n  Total diagnosis records: ~{len(valid)*2} (paired real-time + batch)")
    print(f"  Grand total with existing: ~144")


if __name__ == "__main__":
    main()
