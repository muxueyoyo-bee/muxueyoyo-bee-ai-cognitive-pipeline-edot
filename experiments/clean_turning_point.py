"""
Clean Turning Point Experiment — n=6, empty KB for every run.

Design:
  For each of 6 instances, run TWICE with fresh empty KB:
    Run A (baseline): standard pipeline, no diagnosis
    Run B (real-time): turning point pipeline, Pro diagnosis at A0 REJECT

  This eliminates KB contamination — the only difference between A and B
  is whether real-time external diagnosis is injected at first REJECT.
"""
import json, os, re, sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent_system import DeepSeekLLM, KnowledgeBase, AgentPrompts, PROMPTS_V1, DEAPipeline

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUT_DIR = Path("D:/数据/论文数据/SWE-bench_DEA/clean_turning_point")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Selected instances (same as before)
SELECTED_IDS = [
    "astropy__astropy-14598",
    "astropy__astropy-14369",
    "astropy__astropy-13579",
    "django__django-11400",
    "django__django-11433",
    "django__django-11490",
]


class ExternalDiagnoser:
    """DeepSeek V4 Pro — stronger model for real-time diagnosis."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        self.model = "deepseek-v4-pro"

    def diagnose(self, instance_id: str, problem: str, d_output: str,
                 e_output: str, a_output: str) -> str:
        system_prompt = """You are an **External Diagnoser** — a stronger model called in to rescue a failing bug-fix attempt.

A multi-agent system (Locator -> Fixer -> Reviewer) has just produced a REJECT. Diagnose using the D-E-A framework and give concrete Fixer guidance.

Output format:
```
## D-E-A Diagnosis
D: <was decomposition adequate?>
E: <was evaluation/prioritization justified with explicit criteria?>
A: <what specifically failed in the patch?>

## Root Cause
<single most critical failure>

## Fixer Guidance
<3-5 concrete, numbered instructions for the Fixer to produce a correct patch>
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
            messages=[{"role": "system", "content": system_prompt},
                       {"role": "user", "content": user_prompt}],
            temperature=0.3, max_tokens=2048, stream=False,
        )
        return resp.choices[0].message.content


class CleanTurningPointPipeline(DEAPipeline):
    """Modified pipeline: injects external diagnosis at first REJECT."""

    def __init__(self, llm, prompts, kb, diagnoser, max_revisions=2):
        super().__init__(llm, prompts, kb, max_revisions)
        self.diagnoser = diagnoser

    def run_one(self, instance: dict) -> dict:
        inst_id = instance["instance_id"]
        problem = instance["problem_statement"]
        kb_ctx = self.kb.get_kb_context(max_entries=5)

        print(f"  D...", end="", flush=True)
        d_user = f"{kb_ctx}\n\n## Current Issue\n{problem}"
        d_response = self.llm.chat(self.prompts.locator, d_user)
        subproblems = self._parse_subproblems(d_response)

        print(f"{len(subproblems)} subs E...", end="", flush=True)
        e_user = (f"{kb_ctx}\n\n## Current Issue\n{problem}\n\n"
                  f"## Locator Output\n{d_response}")
        e_responses = [self.llm.chat(self.prompts.fixer, e_user)]
        current_patch = self._extract_patch(e_responses[0])

        revision = 0
        final_verdict = "REJECT"
        a_response = ""
        diagnosis_injected = False
        diagnosis_text = ""
        diagnosis_round = -1

        while revision <= self.max_revisions:
            print(f"A{revision}...", end="", flush=True)
            a_user = (f"{kb_ctx}\n\n## Original Issue\n{problem}\n\n"
                      f"## Fixer Output\n{e_responses[-1]}\n")
            a_response = self.llm.chat(self.prompts.reviewer, a_user)
            verdict = self._parse_verdict(a_response)

            if verdict == "PASS":
                final_verdict = "PASS"
                break
            elif revision < self.max_revisions:
                if not diagnosis_injected:
                    print(f"[DIAG]...", end="", flush=True)
                    try:
                        diagnosis_text = self.diagnoser.diagnose(
                            inst_id, problem, d_response, e_responses[-1], a_response)
                        diagnosis_injected = True
                        diagnosis_round = revision
                        print("OK", end="", flush=True)
                    except Exception as e:
                        print(f"FAIL({e})", end="", flush=True)
                        diagnosis_text = f"[Diagnosis failed: {e}]"

                if diagnosis_text:
                    f_user = (f"{kb_ctx}\n\n## Original Issue\n{problem}\n\n"
                              f"## Reviewer REJECTED your patch\n{a_response}\n\n"
                              f"## [EXTERNAL DIAGNOSIS] (real-time expert feedback)\n{diagnosis_text}\n\n"
                              f"## Your previous output\n{e_responses[-1]}\n\n"
                              f"Revise your patch addressing ALL reviewer concerns AND the external diagnosis.")
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

        return {
            "instance_id": inst_id,
            "repo": instance.get("repo", ""),
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


def load_instances():
    from datasets import load_from_disk
    ds = load_from_disk("D:/数据/论文数据/SWE-bench_Verified")
    lookup = {inst["instance_id"]: inst for inst in list(ds)}
    return [lookup[sid] for sid in SELECTED_IDS if sid in lookup]


def run_clean():
    print("=" * 70)
    print("CLEAN TURNING POINT EXPERIMENT")
    print("  Empty KB for every run. 6 instances x 2 conditions = 12 runs.")
    print("  A = baseline (no diagnosis)  |  B = real-time Pro diagnosis")
    print("=" * 70)

    instances = load_instances()
    print(f"Loaded {len(instances)} instances.\n")

    exec_llm = DeepSeekLLM()       # Flash
    diagnoser = ExternalDiagnoser() # Pro

    results_a = []  # baseline
    results_b = []  # real-time

    for i, inst in enumerate(instances):
        iid = inst["instance_id"]

        # ── Condition A: Baseline (empty KB, no diagnosis) ──
        print(f"[A-{i+1}/6] {iid}", end="")
        kb_a = KnowledgeBase()
        pipe_a = DEAPipeline(exec_llm, PROMPTS_V1, kb_a)
        try:
            ra = pipe_a.run_one(inst)
            ra["condition"] = "A_baseline"
            results_a.append(ra)
            with open(OUT_DIR / f"A_{iid.replace('/', '_')}.json", "w", encoding="utf-8") as f:
                json.dump(ra, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f" FAIL({e})")
            results_a.append({"instance_id": iid, "condition": "A_baseline",
                              "error": str(e), "reviewer_verdict": "ERROR"})

        # ── Condition B: Real-time diagnosis (empty KB) ──
        print(f"  [B-{i+1}/6] {iid}", end="")
        kb_b = KnowledgeBase()
        pipe_b = CleanTurningPointPipeline(exec_llm, PROMPTS_V1, kb_b, diagnoser)
        try:
            rb = pipe_b.run_one(inst)
            rb["condition"] = "B_realtime_diag"
            results_b.append(rb)
            with open(OUT_DIR / f"B_{iid.replace('/', '_')}.json", "w", encoding="utf-8") as f:
                json.dump(rb, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f" FAIL({e})")
            results_b.append({"instance_id": iid, "condition": "B_realtime_diag",
                              "error": str(e), "reviewer_verdict": "ERROR"})

    # ── Comparison Table ──
    print(f"\n{'='*70}")
    print("CLEAN COMPARISON: Baseline vs Real-time Diagnosis (empty KB, n=6)")
    print(f"{'='*70}")
    print(f"{'Instance':<30} {'A_base':<10} {'B_diag':<10} {'Delta':<10} {'B_Diag@':<8}")
    print(f"{'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    a_pass = 0
    b_pass = 0
    b_diag_count = 0
    for i, inst in enumerate(instances):
        iid = inst["instance_id"]
        av = results_a[i].get("reviewer_verdict", "?") if i < len(results_a) else "?"
        bv = results_b[i].get("reviewer_verdict", "?") if i < len(results_b) else "?"
        b_diag = results_b[i].get("diagnosis_injected", False) if i < len(results_b) else False
        b_dround = results_b[i].get("diagnosis_round", -1) if i < len(results_b) else -1

        if av == "PASS": a_pass += 1
        if bv == "PASS": b_pass += 1
        if b_diag: b_diag_count += 1

        delta = ""
        if av != "?" and bv != "?":
            if bv == av: delta = "same"
            elif bv == "PASS" and av != "PASS": delta = "UP"
            else: delta = "DOWN"
        diag_str = f"A{b_dround}" if b_diag else "none"

        print(f"{iid:<30} {av:<10} {bv:<10} {delta:<10} {diag_str:<8}")

    print(f"\n  A (baseline)       PASS: {a_pass}/{len(instances)}")
    print(f"  B (real-time diag) PASS: {b_pass}/{len(instances)}")
    print(f"  Diagnosis injections: {b_diag_count}/{len(instances)}")
    print(f"\n  Results -> {OUT_DIR}")
    print("DONE.")


if __name__ == "__main__":
    run_clean()
