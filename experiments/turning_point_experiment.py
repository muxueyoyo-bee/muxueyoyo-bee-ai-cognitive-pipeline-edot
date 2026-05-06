"""
Turning Point Experiment: Real-time external diagnosis at first REJECT.

Key difference from Phase 1/2/3:
- Instead of batch diagnosis after all 36 instances, the external diagnoser
  intervenes DURING a problematic instance, at the first A0 REJECT.
- This tests whether diagnosis TIMING has an independent effect on outcomes.

Design:
  Execution model: DeepSeek V4 Flash (same as P1/P2/P3)
  External diagnoser: DeepSeek V4 Pro (stronger, simulates Mimo)
  Trigger: first A0 REJECT → send trajectory to Pro → inject feedback → Fixer revises

Selected instances (6 total):
  Old 36: astropy-14598, astropy-14369, astropy-13579
  New 36: django-11400, django-11433, django-11490
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Fix Windows GBK encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Reuse the existing agent system
sys.path.insert(0, str(Path(__file__).parent))
from agent_system import (
    DeepSeekLLM, KnowledgeBase, AgentPrompts, PROMPTS_V1,
    DEAPipeline,
)

OUT_DIR = Path("D:/数据/论文数据/SWE-bench_DEA/turning_point_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# External Diagnoser — stronger model for real-time intervention
# ============================================================

class ExternalDiagnoser:
    """Uses a stronger model (DeepSeek V4 Pro) to diagnose trajectories in real-time."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        self.model = "deepseek-v4-pro"  # stronger than Flash

    def diagnose(self, instance_id: str, problem: str, d_output: str,
                 e_output: str, a_output: str) -> str:
        """Diagnose a failing trajectory and give actionable fix guidance."""

        system_prompt = """You are an **External Diagnoser** — a stronger model called in to rescue a failing bug-fix attempt.

A multi-agent system (Locator→Fixer→Reviewer) has just produced a REJECT on a GitHub issue.
Your job: diagnose WHAT went wrong (D/E/A) and give the Fixer CONCRETE guidance to fix it.

Analyze using the D-E-A framework:
- **D (Decoupling)**: Were sub-problems properly decomposed? Too few? Wrong granularity?
- **E (Evaluation)**: Was priority justified with explicit criteria? Were any sub-problems incorrectly abandoned?
- **A (Aggregation)**: Did the patch actually address the issue? Is the patch complete or truncated?

Output format:
```
## D-E-A Diagnosis
D: <brief — was decomposition adequate?>
E: <brief — was evaluation/prioritization justified?>
A: <brief — what specifically failed in the patch?>

## Root Cause
<single most critical failure>

## Fixer Guidance
<3-5 concrete, specific instructions for the Fixer to produce a correct patch>
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


# ============================================================
# Turning Point Pipeline — real-time diagnosis at first REJECT
# ============================================================

class TurningPointPipeline(DEAPipeline):
    """Modified pipeline: when first REJECT happens, call external diagnoser immediately."""

    def __init__(self, llm: DeepSeekLLM, prompts: AgentPrompts, kb: KnowledgeBase,
                 diagnoser: ExternalDiagnoser, max_revisions: int = 2):
        super().__init__(llm, prompts, kb, max_revisions)
        self.diagnoser = diagnoser
        self.diagnosis_log = []  # track when/where diagnosis was injected

    def run_one(self, instance: dict) -> dict:
        inst_id = instance["instance_id"]
        repo = instance["repo"]
        difficulty = instance.get("difficulty", "unknown")
        problem = instance["problem_statement"]

        kb_ctx = self.kb.get_kb_context(max_entries=5)
        kb_summary = self.kb.get_pattern_summary()
        print(f"  [{inst_id}] ({kb_summary}) D...", end="", flush=True)

        # Step 1: Decoupling
        d_user = (f"{kb_ctx}\n\n## Current Issue\n{problem}")
        d_response = self.llm.chat(self.prompts.locator, d_user)
        subproblems = self._parse_subproblems(d_response)
        self.kb.add("Locator", inst_id, "subproblems", json.dumps(subproblems, ensure_ascii=False))
        print(f" {len(subproblems)} subs → E...", end="", flush=True)

        # Step 2: Evaluation → Fix
        e_user = (f"{kb_ctx}\n\n"
                  f"## Current Issue\n{problem}\n\n"
                  f"## Locator Output\n{d_response}")
        e_responses = [self.llm.chat(self.prompts.fixer, e_user)]
        current_patch = self._extract_patch(e_responses[0])

        # Step 3: Aggregation Review — with real-time diagnosis at first REJECT
        revision = 0
        final_verdict = "REJECT"
        a_response = ""
        diagnosis_injected = False
        diagnosis_text = ""
        diagnosis_round = -1

        while revision <= self.max_revisions:
            print(f" A{revision}...", end="", flush=True)
            a_user = (f"{kb_ctx}\n\n"
                      f"## Original Issue\n{problem}\n\n"
                      f"## Fixer Output\n{e_responses[-1]}\n")
            a_response = self.llm.chat(self.prompts.reviewer, a_user)
            verdict = self._parse_verdict(a_response)

            if verdict == "PASS":
                final_verdict = "PASS"
                break
            elif revision < self.max_revisions:
                # ---- TURNING POINT: first REJECT triggers external diagnosis ----
                if not diagnosis_injected:
                    print(f" [DIAG]...", end="", flush=True)
                    try:
                        diagnosis_text = self.diagnoser.diagnose(
                            inst_id, problem, d_response,
                            e_responses[-1], a_response
                        )
                        diagnosis_injected = True
                        diagnosis_round = revision
                        self.diagnosis_log.append({
                            "instance_id": inst_id,
                            "round": revision,
                            "diagnosis": diagnosis_text[:1500],
                            "timestamp": datetime.now().isoformat(),
                        })
                        print(f" OK", end="", flush=True)
                    except Exception as e:
                        print(f" FAIL({e})", end="", flush=True)
                        diagnosis_text = f"[Diagnosis failed: {e}]"

                # Build revision prompt — inject diagnosis if available
                if diagnosis_text:
                    f_user = (f"{kb_ctx}\n\n"
                              f"## Original Issue\n{problem}\n\n"
                              f"## Reviewer REJECTED your patch\n{a_response}\n\n"
                              f"## [EXTERNAL DIAGNOSIS] (real-time expert feedback)\n"
                              f"{diagnosis_text}\n\n"
                              f"## Your previous output\n{e_responses[-1]}\n\n"
                              f"Revise your patch addressing ALL reviewer concerns "
                              f"AND the external diagnosis guidance.")
                else:
                    f_user = (f"{kb_ctx}\n\n"
                              f"## Original Issue\n{problem}\n\n"
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
            "instance_id": inst_id,
            "repo": repo,
            "difficulty": difficulty,
            "num_subproblems": len(subproblems),
            "subproblems": subproblems,
            "reviewer_verdict": final_verdict,
            "revision_count": revision,
            "patch_chars": len(current_patch),
            "flags": flags,
            "diagnosis_injected": diagnosis_injected,
            "diagnosis_round": diagnosis_round,
            "diagnosis_preview": diagnosis_text[:500] if diagnosis_text else "",
            "trajectory": {
                "instance_id": inst_id,
                "repo": repo,
                "difficulty": difficulty,
                "problem_statement": problem,
                "D_output": d_response,
                "E_rounds": e_responses,
                "A_response": a_response,
                "final_verdict": final_verdict,
                "final_patch": current_patch[:2000],
                "diagnosis_injected": diagnosis_injected,
                "diagnosis_text": diagnosis_text[:2000],
            }
        }


# ============================================================
# Experiment Runner
# ============================================================

def load_selected_instances() -> list[dict]:
    """Load the 6 selected instances from both old and new sets."""
    from datasets import load_from_disk
    ds = load_from_disk("D:/数据/论文数据/SWE-bench_Verified")
    all_instances = list(ds)

    # Build lookup by instance_id
    lookup = {}
    for inst in all_instances:
        lookup[inst["instance_id"]] = inst

    selected_ids = [
        # Old 36 (indices 0-35)
        "astropy__astropy-14598",   # P3 still REJECT — extreme case
        "astropy__astropy-14369",   # P1 PASS→P2 REJECT→P3 PASS
        "astropy__astropy-13579",   # P1 REJECT→P2 REJECT→P3 PASS
        # New 36 (indices 36-71)
        "django__django-11400",     # REJECT cluster start
        "django__django-11433",     # 0 subproblems — D崩溃
        "django__django-11490",     # REJECT cluster end
    ]

    instances = []
    for sid in selected_ids:
        if sid in lookup:
            instances.append(lookup[sid])
        else:
            print(f"  WARNING: {sid} not found in dataset!")

    return instances


def run_turning_point_experiment():
    print("=" * 70)
    print("TURNING POINT EXPERIMENT: Real-time External Diagnosis at First REJECT")
    print("=" * 70)
    print(f"  Execution: DeepSeek V4 Flash")
    print(f"  Diagnoser: DeepSeek V4 Pro (real-time, at A0 REJECT)")
    print(f"  vs P1 (baseline) / P2 (self-reflection) / P3 (batch Mimo)")
    print(f"  Output: {OUT_DIR}")
    print()

    # Load instances
    instances = load_selected_instances()
    print(f"Loaded {len(instances)} instances for turning point experiment.\n")

    # Init
    exec_llm = DeepSeekLLM()  # Flash
    diagnoser = ExternalDiagnoser()  # Pro
    kb = KnowledgeBase(persistent_file=str(OUT_DIR / "turning_point_kb.json"))
    pipeline = TurningPointPipeline(exec_llm, PROMPTS_V1, kb, diagnoser)

    # Run
    results = []
    for i, inst in enumerate(instances):
        inst_id = inst["instance_id"]
        print(f"[TP] {i+1}/{len(instances)}", end="")
        try:
            result = pipeline.run_one(inst)
            result["phase"] = "turning_point"
            result["prompt_version"] = PROMPTS_V1.version
            results.append(result)

            traj_file = OUT_DIR / f"{inst_id.replace('/', '_')}.json"
            with open(traj_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            kb.save(str(OUT_DIR / "turning_point_kb.json"))
        except Exception as e:
            print(f"  !! Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "instance_id": inst_id, "phase": "turning_point", "error": str(e),
                "flags": ["EXEC_ERROR"], "reviewer_verdict": "ERROR"
            })

    # Save diagnosis log
    with open(OUT_DIR / "diagnosis_log.json", "w", encoding="utf-8") as f:
        json.dump(pipeline.diagnosis_log, f, ensure_ascii=False, indent=2)

    # ================================================================
    # Cross-method comparison
    # ================================================================
    print(f"\n{'='*70}")
    print("CROSS-METHOD COMPARISON: Turning Point vs P1/P2/P3")
    print(f"{'='*70}")

    # Load existing results for the same instances
    existing = {}
    for phase, dir_name in [("P1_old", "phase1_baseline"), ("P2", "phase2_self_reflection"),
                              ("P3", "phase3_old36"), ("P1_new", "phase1_new36")]:
        phase_dir = Path(f"D:/数据/论文数据/SWE-bench_DEA/{dir_name}")
        if phase_dir.exists():
            for f in phase_dir.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    iid = data.get("instance_id", "")
                    if iid in [inst["instance_id"] for inst in instances]:
                        existing.setdefault(iid, {})[phase] = data
                except:
                    pass

    print(f"\n{'Instance':<30} {'P1':<6} {'P2':<6} {'P3':<6} {'TP':<6} {'Δ(P3→TP)':<10} {'Diag@':<8}")
    print(f"{'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*8}")

    comparison_data = {}
    for inst in instances:
        iid = inst["instance_id"]
        tp_result = next((r for r in results if r.get("instance_id") == iid), None)
        tp_verdict = tp_result.get("reviewer_verdict", "?") if tp_result else "?"
        tp_revs = tp_result.get("revision_count", "?") if tp_result else "?"
        tp_diag = tp_result.get("diagnosis_injected", False) if tp_result else False
        tp_dround = tp_result.get("diagnosis_round", -1) if tp_result else -1

        ex = existing.get(iid, {})
        p1_v = ex.get("P1_old", ex.get("P1_new", {})).get("final_verdict", "?")
        p2_v = ex.get("P2", {}).get("final_verdict", "?")
        p3_v = ex.get("P3", {}).get("final_verdict", "?")

        delta = ""
        if p3_v != "?" and tp_verdict != "?":
            if p3_v == tp_verdict:
                delta = "same"
            elif tp_verdict == "PASS" and p3_v == "REJECT":
                delta = "↑IMPROVED"
            elif tp_verdict == "REJECT" and p3_v == "PASS":
                delta = "↓WORSE"

        diag_str = f"A{tp_dround}" if tp_diag else "none"

        print(f"{iid:<30} {p1_v:<6} {p2_v:<6} {p3_v:<6} {tp_verdict:<6} {delta:<10} {diag_str:<8}")

        comparison_data[iid] = {
            "P1": p1_v, "P2": p2_v, "P3": p3_v, "TP": tp_verdict,
            "TP_revisions": tp_revs, "TP_diagnosis_round": tp_dround,
            "TP_diagnosis_injected": tp_diag,
        }

    # Summary stats
    tp_passes = sum(1 for r in results if r.get("reviewer_verdict") == "PASS")
    tp_total = len([r for r in results if not r.get("error")])
    print(f"\n  TP PASS: {tp_passes}/{tp_total}")
    print(f"  Diagnosis injections: {sum(1 for r in results if r.get('diagnosis_injected'))}")

    # Save comparison
    with open(OUT_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)

    print(f"\n  Results → {OUT_DIR}")
    print(f"  Comparison → {OUT_DIR / 'comparison.json'}")
    print("DONE.")


if __name__ == "__main__":
    run_turning_point_experiment()
