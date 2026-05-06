"""
Turning Point Validation Experiment: 72-instance, 2-condition real-time diagnosis.

Expands the pilot (n=6×2) to full validation:
  Condition 1: Real-time diagnosis at first REJECT (immediate intervention)
  Condition 2: Self-reflection buffer → real-time diagnosis only if still REJECT

Design:
  Execution model: DeepSeek V4 Flash (same as P1/P2/P3)
  External diagnoser: DeepSeek V4 Pro
  Self-reflector: DeepSeek V4 Flash (Meta-Analyst, per-instance)

Instances:
  Old 36: SWE-bench Verified indices 0-35 (astropy + django)
  New 36: SWE-bench Verified indices 36-71 (all django)

Total: 72 instances × 2 conditions = up to 144 runs
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
BASE_OUT = Path("D:/数据/论文数据/SWE-bench_DEA/turning_point_validation")
COND1_DIR = BASE_OUT / "condition1_realtime_diag"
COND2_DIR = BASE_OUT / "condition2_selfreflect_then_diag"
for d in [BASE_OUT, COND1_DIR, COND2_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Instance loading ────────────────────────────────────────────

def load_all_72_instances() -> list[dict]:
    """Load old 36 (indices 0-35) + new 36 (indices 36-71) from dataset."""
    from datasets import load_from_disk
    ds = load_from_disk("D:/数据/论文数据/SWE-bench_Verified")
    all_instances = list(ds)

    old_36 = all_instances[:36]   # indices 0-35  (astropy + django)
    new_36 = all_instances[36:72]  # indices 36-71 (all django)

    for inst in old_36:
        inst["_set"] = "old36"
    for inst in new_36:
        inst["_set"] = "new36"

    instances = old_36 + new_36
    print(f"Loaded {len(old_36)} old + {len(new_36)} new = {len(instances)} total instances.")
    return instances


# ── External Diagnoser (same as pilot) ───────────────────────────

class ExternalDiagnoser:
    """Uses DeepSeek V4 Pro to diagnose trajectories in real-time."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", "REDACTED"),
            base_url="https://api.deepseek.com",
        )
        self.model = "deepseek-v4-pro"

    def diagnose(self, instance_id: str, problem: str, d_output: str,
                 e_output: str, a_output: str) -> str:
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


# ── Per-instance Self-Reflector ─────────────────────────────────

class PerInstanceReflector:
    """On a REJECT, runs a quick self-reflection on the current instance's trajectory
    to see if the system can fix itself before calling the external diagnoser."""

    def __init__(self):
        self.llm = DeepSeekLLM()  # Flash — same as execution model

    def reflect(self, instance_id: str, problem: str, d_output: str,
                e_output: str, a_output: str) -> str:
        """Analyze the failure and give the Fixer concrete revision guidance.
        This is SELF-reflection — the same model analyzing its own failure."""

        system_prompt = """You are a **Self-Reflection Analyst**.
The multi-agent system you are part of has just produced a REJECT on a GitHub issue.

Your job: look at what went wrong and produce concrete, actionable fix guidance for the Fixer Agent.
Focus on what the Fixer should DO differently — not abstract criticism.

Be specific: "add a null check before line 45" not "be more careful about edge cases."
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

Analyze the failure and give the Fixer specific revision instructions.
Output format:
```
## What Went Wrong
<brief root cause>

## Fixer Revision Instructions
<3-5 concrete, specific instructions>
```
"""
        resp = self.llm.chat(system_prompt, user_prompt, temperature=0.3)
        return resp


# ── Condition 1 Pipeline: Real-time diagnosis at first REJECT ────

class Condition1Pipeline(DEAPipeline):
    """When first REJECT happens, call external diagnoser immediately."""

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

        # Step 1: Decoupling
        d_user = f"{kb_ctx}\n\n## Current Issue\n{problem}"
        d_response = self.llm.chat(self.prompts.locator, d_user)
        subproblems = self._parse_subproblems(d_response)
        self.kb.add("Locator", inst_id, "subproblems", json.dumps(subproblems, ensure_ascii=False))
        print(f" {len(subproblems)} subs → E...", end="", flush=True)

        # Step 2: Evaluation → Fix
        e_user = (f"{kb_ctx}\n\n## Current Issue\n{problem}\n\n"
                  f"## Locator Output\n{d_response}")
        e_responses = [self.llm.chat(self.prompts.fixer, e_user)]
        current_patch = self._extract_patch(e_responses[0])

        # Step 3: Aggregation Review — with CONDITION 1 diagnosis at first REJECT
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
                # ── CONDITION 1: first REJECT → immediate external diagnosis ──
                if not diag_injected:
                    print(f" [EXT-DIAG]...", end="", flush=True)
                    try:
                        diag_text = self.diagnoser.diagnose(
                            inst_id, problem, d_response,
                            e_responses[-1], a_response
                        )
                        diag_injected = True
                        diag_round = revision
                        self.diag_log.append({
                            "instance_id": inst_id, "condition": "C1",
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
                              f"## [EXTERNAL DIAGNOSIS] (real-time expert feedback)\n"
                              f"{diag_text}\n\n"
                              f"## Your previous output\n{e_responses[-1]}\n\n"
                              f"Revise your patch addressing ALL reviewer concerns "
                              f"AND the external diagnosis guidance.")
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
            "condition": "C1_realtime_diag",
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


# ── Condition 2 Pipeline: Self-reflection first, then diagnosis ──

class Condition2Pipeline(DEAPipeline):
    """On first REJECT: self-reflection first. Only if still REJECT after reflection
    revision, THEN trigger external diagnosis."""

    def __init__(self, llm: DeepSeekLLM, prompts: AgentPrompts, kb: KnowledgeBase,
                 diagnoser: ExternalDiagnoser, reflector: PerInstanceReflector,
                 max_revisions: int = 2):
        super().__init__(llm, prompts, kb, max_revisions)
        self.diagnoser = diagnoser
        self.reflector = reflector
        self.diag_log = []

    def run_one(self, instance: dict) -> dict:
        inst_id = instance["instance_id"]
        repo = instance["repo"]
        problem = instance["problem_statement"]
        instance_set = instance.get("_set", "?")

        kb_ctx = self.kb.get_kb_context(max_entries=5)
        kb_summary = self.kb.get_pattern_summary()
        print(f"  [{inst_id}] ({kb_summary}) D...", end="", flush=True)

        # Step 1: Decoupling
        d_user = f"{kb_ctx}\n\n## Current Issue\n{problem}"
        d_response = self.llm.chat(self.prompts.locator, d_user)
        subproblems = self._parse_subproblems(d_response)
        self.kb.add("Locator", inst_id, "subproblems", json.dumps(subproblems, ensure_ascii=False))
        print(f" {len(subproblems)} subs → E...", end="", flush=True)

        # Step 2: Evaluation → Fix
        e_user = (f"{kb_ctx}\n\n## Current Issue\n{problem}\n\n"
                  f"## Locator Output\n{d_response}")
        e_responses = [self.llm.chat(self.prompts.fixer, e_user)]
        current_patch = self._extract_patch(e_responses[0])

        # Step 3: Review — CONDITION 2 logic
        revision = 0
        final_verdict = "REJECT"
        a_response = ""
        self_reflection_done = False
        self_reflection_text = ""
        self_reflection_round = -1
        self_reflection_helped = False
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
                # ── CONDITION 2: first REJECT → self-reflection first ──
                if not self_reflection_done:
                    print(f" [SELF-REFLECT]...", end="", flush=True)
                    try:
                        self_reflection_text = self.reflector.reflect(
                            inst_id, problem, d_response,
                            e_responses[-1], a_response
                        )
                        self_reflection_done = True
                        self_reflection_round = revision
                        print(f" OK", end="", flush=True)
                    except Exception as e:
                        print(f" FAIL({e})", end="", flush=True)
                        self_reflection_text = f"[Self-reflection failed: {e}]"

                    # Apply self-reflection guidance to revision
                    f_user = (f"{kb_ctx}\n\n## Original Issue\n{problem}\n\n"
                              f"## Reviewer REJECTED your patch\n{a_response}\n\n"
                              f"## [SELF-REFLECTION ANALYSIS]\n{self_reflection_text}\n\n"
                              f"## Your previous output\n{e_responses[-1]}\n\n"
                              f"Revise your patch addressing ALL reviewer concerns "
                              f"AND the self-reflection guidance.")
                    f_response = self.llm.chat(self.prompts.fixer, f_user, temperature=0.4)
                    current_patch = self._extract_patch(f_response)
                    e_responses.append(f"--- Revision {revision+1} (after self-reflection) ---\n{f_response}")
                    revision += 1

                    # Don't check inline — let next loop iteration's reviewer
                    # naturally evaluate the post-SR fixer output.
                    # If still REJECT, the loop will reach the external diagnosis branch below.
                    continue

                # ── Still failing after self-reflection → external diagnosis ──
                if self_reflection_done and not diag_injected:
                    print(f" [EXT-DIAG(post-reflect)]...", end="", flush=True)
                    try:
                        diag_text = self.diagnoser.diagnose(
                            inst_id, problem, d_response,
                            e_responses[-1], a_response
                        )
                        diag_injected = True
                        diag_round = revision
                        self.diag_log.append({
                            "instance_id": inst_id, "condition": "C2",
                            "round": revision,
                            "self_reflection_round": self_reflection_round,
                            "diagnosis": diag_text[:1500],
                            "timestamp": datetime.now().isoformat(),
                        })
                        print(f" OK", end="", flush=True)
                    except Exception as e:
                        print(f" FAIL({e})", end="", flush=True)
                        diag_text = f"[Diagnosis failed: {e}]"

                if diag_text:
                    f_user = (f"{kb_ctx}\n\n## Original Issue\n{problem}\n\n"
                              f"## Reviewer REJECTED your patch\n{a_response}\n\n"
                              f"## [EXTERNAL DIAGNOSIS] (real-time expert feedback)\n"
                              f"{diag_text}\n\n"
                              f"## Your previous output\n{e_responses[-1]}\n\n"
                              f"Revise your patch addressing ALL reviewer concerns "
                              f"AND the external diagnosis guidance.")
                else:
                    f_user = (f"{kb_ctx}\n\n## Original Issue\n{problem}\n\n"
                              f"## Reviewer REJECTED your patch\n{a_response}\n\n"
                              f"## Your previous output\n{e_responses[-1]}\n\n"
                              f"Revise your patch addressing ALL reviewer concerns.")

                f_response = self.llm.chat(self.prompts.fixer, f_user, temperature=0.4)
                current_patch = self._extract_patch(f_response)
                e_responses.append(f"--- Revision {revision+1} (external diag) ---\n{f_response}")
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
            "self_reflection_done": self_reflection_done,
            "self_reflection_helped": self_reflection_helped,
            "self_reflection_round": self_reflection_round,
            "diagnosis_injected": diag_injected, "diagnosis_round": diag_round,
            "diagnosis_preview": diag_text[:500] if diag_text else "",
            "condition": "C2_selfreflect_then_diag",
            "trajectory": {
                "instance_id": inst_id, "repo": repo,
                "problem_statement": problem,
                "D_output": d_response, "E_rounds": e_responses,
                "A_response": a_response, "final_verdict": final_verdict,
                "final_patch": current_patch[:2000],
                "self_reflection_done": self_reflection_done,
                "self_reflection_text": self_reflection_text[:2000],
                "diagnosis_injected": diag_injected,
                "diagnosis_text": diag_text[:2000],
            }
        }


# ── Run Condition 1 ─────────────────────────────────────────────

def run_condition1(instances: list[dict], start_idx: int = 0):
    """Condition 1: Real-time external diagnosis at first REJECT."""
    print("=" * 70)
    print("CONDITION 1: Real-time External Diagnosis at First REJECT")
    print(f"  Execution: DeepSeek V4 Flash")
    print(f"  Diagnoser: DeepSeek V4 Pro (immediate on A0 REJECT)")
    print(f"  Instances: {len(instances)} (indices {start_idx}+)")
    print(f"  Output: {COND1_DIR}")
    print("=" * 70)

    exec_llm = DeepSeekLLM()
    diagnoser = ExternalDiagnoser()
    kb = KnowledgeBase(persistent_file=str(COND1_DIR / "c1_kb.json"))
    pipeline = Condition1Pipeline(exec_llm, PROMPTS_V1, kb, diagnoser)

    results = []
    for i, inst in enumerate(instances):
        if i < start_idx:
            continue
        inst_id = inst["instance_id"]
        print(f"[C1] {i+1}/{len(instances)}", end="")
        try:
            result = pipeline.run_one(inst)
            result["global_index"] = i
            results.append(result)

            traj_file = COND1_DIR / f"{inst_id.replace('/', '_')}.json"
            with open(traj_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # Save checkpoint every 5 instances
            if (i + 1) % 5 == 0:
                kb.save(str(COND1_DIR / "c1_kb.json"))
                with open(COND1_DIR / "c1_checkpoint.json", "w", encoding="utf-8") as f:
                    json.dump({"last_idx": i, "n_results": len(results)}, f)
                print(f"  [checkpoint saved @ {i+1}]")
        except Exception as e:
            print(f"  !! Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "instance_id": inst_id, "condition": "C1", "global_index": i,
                "error": str(e), "flags": ["EXEC_ERROR"], "reviewer_verdict": "ERROR"
            })

    # Save final
    kb.save(str(COND1_DIR / "c1_kb.json"))
    with open(COND1_DIR / "c1_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(COND1_DIR / "c1_diag_log.json", "w", encoding="utf-8") as f:
        json.dump(pipeline.diag_log, f, ensure_ascii=False, indent=2)

    return results


# ── Run Condition 2 ─────────────────────────────────────────────

def run_condition2(instances: list[dict], start_idx: int = 0):
    """Condition 2: Self-reflection first on REJECT, external diagnosis only if still stuck."""
    print("\n" + "=" * 70)
    print("CONDITION 2: Self-Reflection Buffer → External Diagnosis (if still REJECT)")
    print(f"  Execution: DeepSeek V4 Flash")
    print(f"  Self-reflector: DeepSeek V4 Flash (per-instance)")
    print(f"  Diagnoser: DeepSeek V4 Pro (only if self-reflection fails)")
    print(f"  Instances: {len(instances)} (indices {start_idx}+)")
    print(f"  Output: {COND2_DIR}")
    print("=" * 70)

    exec_llm = DeepSeekLLM()
    diagnoser = ExternalDiagnoser()
    reflector = PerInstanceReflector()
    kb = KnowledgeBase(persistent_file=str(COND2_DIR / "c2_kb.json"))
    pipeline = Condition2Pipeline(exec_llm, PROMPTS_V1, kb, diagnoser, reflector)

    results = []
    for i, inst in enumerate(instances):
        if i < start_idx:
            continue
        inst_id = inst["instance_id"]
        print(f"[C2] {i+1}/{len(instances)}", end="")
        try:
            result = pipeline.run_one(inst)
            result["global_index"] = i
            results.append(result)

            traj_file = COND2_DIR / f"{inst_id.replace('/', '_')}.json"
            with open(traj_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            if (i + 1) % 5 == 0:
                kb.save(str(COND2_DIR / "c2_kb.json"))
                with open(COND2_DIR / "c2_checkpoint.json", "w", encoding="utf-8") as f:
                    json.dump({"last_idx": i, "n_results": len(results)}, f)
                print(f"  [checkpoint saved @ {i+1}]")
        except Exception as e:
            print(f"  !! Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "instance_id": inst_id, "condition": "C2", "global_index": i,
                "error": str(e), "flags": ["EXEC_ERROR"], "reviewer_verdict": "ERROR"
            })

    kb.save(str(COND2_DIR / "c2_kb.json"))
    with open(COND2_DIR / "c2_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(COND2_DIR / "c2_diag_log.json", "w", encoding="utf-8") as f:
        json.dump(pipeline.diag_log, f, ensure_ascii=False, indent=2)

    return results


# ── Cross-condition comparison ───────────────────────────────────

def build_comparison(c1_results: list[dict], c2_results: list[dict]):
    """Compare Condition 1 vs Condition 2 vs existing phases."""
    print("\n" + "=" * 70)
    print("CROSS-CONDITION COMPARISON")
    print("=" * 70)

    # Load existing P1/P2/P3 results for cross-reference
    existing = {}
    for phase, dir_name in [("P1_old", "phase1_baseline"), ("P1_new", "phase1_new36"),
                              ("P2", "phase2_self_reflection"), ("P3", "phase3_old36")]:
        phase_dir = Path(f"D:/数据/论文数据/SWE-bench_DEA/{dir_name}")
        if phase_dir.exists():
            for f in phase_dir.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    iid = data.get("instance_id", "")
                    existing.setdefault(iid, {})[phase] = data
                except:
                    pass

    # Also load P1 new36 summaries
    for summary_file in ["phase1_new36_summary.json"]:
        sp = Path(f"D:/数据/论文数据/SWE-bench_DEA/{summary_file}")
        if sp.exists():
            try:
                with open(sp, "r", encoding="utf-8") as f:
                    new36_data = json.load(f)
                # new36 summary may have different format; try to map
                if isinstance(new36_data, list):
                    for item in new36_data:
                        iid = item.get("instance_id", "")
                        if iid:
                            existing.setdefault(iid, {})["P1_new"] = item
            except:
                pass

    # Build lookup
    c1_lookup = {r["instance_id"]: r for r in c1_results if not r.get("error")}
    c2_lookup = {r["instance_id"]: r for r in c2_results if not r.get("error")}

    all_ids = sorted(set(
        [r["instance_id"] for r in c1_results + c2_results if not r.get("error")]
    ))

    # Print header
    print(f"\n{'Instance':<32} {'Set':<6} {'P1':<6} {'P2':<6} {'P3':<6} {'C1':<6} {'C2':<6} {'C1vC2':<10} {'SR_help':<8}")
    print(f"{'-'*32} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*8}")

    c1_passes = 0
    c2_passes = 0
    c1_diags = 0
    c2_diags = 0
    c2_sr_helps = 0
    c2_sr_total = 0
    total = 0

    for iid in all_ids:
        c1 = c1_lookup.get(iid, {})
        c2 = c2_lookup.get(iid, {})
        ex = existing.get(iid, {})

        c1_v = c1.get("reviewer_verdict", "?")
        c2_v = c2.get("reviewer_verdict", "?")
        inst_set = c1.get("_set", c2.get("_set", "?"))

        p1_v = ex.get("P1_old", ex.get("P1_new", ex.get("P1_new", {}))).get(
            "final_verdict", ex.get("P1_old", ex.get("P1_new", ex.get("P1_new", {}))).get(
                "reviewer_verdict", "?"))
        p2_v = ex.get("P2", {}).get("final_verdict", ex.get("P2", {}).get("reviewer_verdict", "?"))
        p3_v = ex.get("P3", {}).get("final_verdict", ex.get("P3", {}).get("reviewer_verdict", "?"))

        # C1 vs C2 comparison
        c1c2 = ""
        if c1_v != "?" and c2_v != "?":
            if c1_v == c2_v:
                c1c2 = "same"
            elif c2_v == "PASS" and c1_v == "REJECT":
                c1c2 = "C2↑BETTER"
            elif c1_v == "PASS" and c2_v == "REJECT":
                c1c2 = "C1↑BETTER"

        sr_help = ""
        if c2.get("self_reflection_done"):
            c2_sr_total += 1
            if c2.get("self_reflection_helped"):
                sr_help = "✓HELPED"
                c2_sr_helps += 1
            else:
                sr_help = "✗no"

        print(f"{iid:<32} {inst_set:<6} {p1_v:<6} {p2_v:<6} {p3_v:<6} {c1_v:<6} {c2_v:<6} {c1c2:<10} {sr_help:<8}")

        total += 1
        if c1_v == "PASS": c1_passes += 1
        if c2_v == "PASS": c2_passes += 1
        if c1.get("diagnosis_injected"): c1_diags += 1
        if c2.get("diagnosis_injected"): c2_diags += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY (n={total} valid instances)")
    print(f"{'='*70}")
    print(f"  Condition 1 (real-time diag):  PASS={c1_passes}/{total} ({100*c1_passes/total:.1f}%) | Diag injections={c1_diags}")
    print(f"  Condition 2 (self-reflect→diag): PASS={c2_passes}/{total} ({100*c2_passes/total:.1f}%) | Diag injections={c2_diags}")
    print(f"  Self-reflection buffer:")
    print(f"    Instances where SR was triggered: {c2_sr_total}")
    print(f"    SR alone fixed the issue:         {c2_sr_helps} ({100*c2_sr_helps/max(c2_sr_total,1):.1f}%)")
    print(f"    SR failed → escalated to diag:    {c2_diags}")
    print(f"  Δ(C2 - C1): {c2_passes - c1_passes:+d} PASS ({100*(c2_passes-c1_passes)/total:+.1f}pp)")

    comparison = {
        "n_instances": total,
        "condition1": {"passes": c1_passes, "pass_rate": c1_passes/total,
                       "diag_injections": c1_diags},
        "condition2": {"passes": c2_passes, "pass_rate": c2_passes/total,
                       "diag_injections": c2_diags,
                       "sr_triggered": c2_sr_total,
                       "sr_alone_fixed": c2_sr_helps,
                       "sr_fix_rate": c2_sr_helps/max(c2_sr_total, 1)},
        "delta_c2_minus_c1": c2_passes - c1_passes,
    }

    with open(BASE_OUT / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"\n  Comparison → {BASE_OUT / 'comparison.json'}")
    return comparison


# ── Main ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Turning Point Validation Experiment")
    parser.add_argument("--condition", choices=["c1", "c2", "both"], default="both",
                       help="Which condition(s) to run")
    parser.add_argument("--start", type=int, default=0,
                       help="Start from instance index (for resume)")
    parser.add_argument("--limit", type=int, default=72,
                       help="Max instances to run")
    parser.add_argument("--compare-only", action="store_true",
                       help="Only build comparison from existing results")
    args = parser.parse_args()

    if args.compare_only:
        # Load existing results
        c1_results = []
        c2_results = []
        for f in COND1_DIR.glob("*.json"):
            if "checkpoint" not in f.name and "kb" not in f.name and "summary" not in f.name and "diag_log" not in f.name:
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        c1_results.append(json.load(fp))
                except: pass
        for f in COND2_DIR.glob("*.json"):
            if "checkpoint" not in f.name and "kb" not in f.name and "summary" not in f.name and "diag_log" not in f.name:
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        c2_results.append(json.load(fp))
                except: pass
        if c1_results and c2_results:
            build_comparison(c1_results, c2_results)
        else:
            print("No existing results found. Run experiment first.")
        return

    instances = load_all_72_instances()
    instances = instances[args.start:args.start + args.limit]
    print(f"Running on {len(instances)} instances (indices {args.start} to {args.start + len(instances) - 1})")

    c1_results = []
    c2_results = []

    if args.condition in ("c1", "both"):
        c1_results = run_condition1(instances)

    if args.condition in ("c2", "both"):
        c2_results = run_condition2(instances)

    if c1_results and c2_results:
        build_comparison(c1_results, c2_results)

    print("\nDONE.")


if __name__ == "__main__":
    main()
