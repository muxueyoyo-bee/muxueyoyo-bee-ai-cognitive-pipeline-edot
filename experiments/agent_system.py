"""
Minimal 3-Agent system for D-E-A cross-scene validation on SWE-bench.

QClaw control-variable design, replicated:
  Phase 1 (Baseline)      : 3 agents run raw → trajectories
  Phase 2 (Self-reflection): Agents meta-analyze Phase 1 trajectories → self-improve → re-run
  Phase 3 (Mimo external) : Mimo diagnoses Phase 2 → external feedback → re-run

Comparison anchors:
  Phase 1 vs Phase 2 = can the system self-diagnose D-E-A defects?
  Phase 2 vs Phase 3 = does external diagnosis (Mimo) add value beyond self-reflection?
  Phase 1 vs Phase 3 = total improvement from baseline to post-intervention

QClaw mapping:
  Phase 1 = Auto时期 (V1.0)
  Phase 2 = DeepSeekV4Pro self-iteration (V2.0 pre-diagnosis)
  Phase 3 = DeepSeek external diagnosis → V2.0 final
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ============================================================
# Knowledge Base — shared across instances, mirrors QClaw's KB
# ============================================================

@dataclass
class KnowledgeBase:
    """Accumulates across instances. Feeds context to each agent call."""
    entries: list[dict] = field(default_factory=list)
    instance_count: int = 0
    persistent_file: str = ""

    def add(self, agent: str, inst_id: str, key: str, value: str):
        self.entries.append({
            "agent": agent, "instance": inst_id, "key": key, "value": value,
            "timestamp": datetime.now().isoformat()
        })

    def add_instance_summary(self, inst_id: str, subs: int, verdict: str,
                              revs: int, flags: list[str], d_output: str):
        """Called after each instance completes."""
        self.instance_count += 1
        self.entries.append({
            "agent": "System",
            "instance": inst_id,
            "key": f"instance_{self.instance_count}_summary",
            "value": {
                "num_subproblems": subs,
                "verdict": verdict,
                "revisions": revs,
                "flags": flags,
                "locator_preview": d_output[:500],
            },
            "timestamp": datetime.now().isoformat()
        })

    def get_kb_context(self, max_entries: int = 5) -> str:
        """Context string injected into each agent call. Shows recent KB entries
        so later instances benefit from earlier instances' patterns."""
        if not self.entries:
            return ""
        recent = self.entries[-max_entries:]
        lines = ["## Knowledge Base (accumulated from previous instances)"]
        for e in recent:
            if isinstance(e.get("value"), dict):
                v = json.dumps(e["value"], ensure_ascii=False)[:400]
            else:
                v = str(e.get("value", ""))[:400]
            lines.append(f"- [{e['agent']}] {e['key']}: {v}")
        return "\n".join(lines)

    def get_pattern_summary(self) -> str:
        """Aggregate pattern summary for Meta-Analyst."""
        if not self.entries:
            return "No KB entries."
        summaries = [e for e in self.entries if "summary" in e.get("key", "")]
        all_flags = []
        for s in summaries:
            v = s.get("value", {})
            if isinstance(v, dict):
                all_flags.extend(v.get("flags", []))
        d_count = sum(1 for f in all_flags if f.startswith("D_"))
        e_count = sum(1 for f in all_flags if f.startswith("E_"))
        a_count = sum(1 for f in all_flags if f.startswith("A_"))
        pass_count = sum(1 for s in summaries
                        if isinstance(s.get("value"), dict)
                        and s["value"].get("verdict") == "PASS")
        return (f"KB: {len(summaries)} instances | PASS={pass_count} | "
                f"Flag distribution: D={d_count} E={e_count} A={a_count} | "
                f"E/(D+A)={e_count/max(d_count+a_count,1):.2f}")

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"instance_count": self.instance_count, "entries": self.entries},
                      f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        kb = cls(entries=data.get("entries", []),
                 instance_count=data.get("instance_count", 0),
                 persistent_file=filepath)
        return kb

# ============================================================
# LLM Backend
# ============================================================

class DeepSeekLLM:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", "REDACTED"),
            base_url="https://api.deepseek.com",
        )
        self.model = os.environ.get("DEEPSEEK_EXEC_MODEL", "deepseek-v4-flash")

    def chat(self, system_prompt: str, user_message: str, temperature: float = 0.3) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=4096,
            stream=False,
        )
        return resp.choices[0].message.content


# ============================================================
# Prompts — versioned, can be updated between phases
# ============================================================

@dataclass
class AgentPrompts:
    """Prompts that evolve across phases. Phase 2 updates from self-reflection,
       Phase 3 updates from Mimo external diagnosis."""
    locator: str
    fixer: str
    reviewer: str
    version: str = "v1.0_baseline"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "locator": self.locator,
            "fixer": self.fixer,
            "reviewer": self.reviewer,
        }


PROMPTS_V1 = AgentPrompts(
    version="v1.0_baseline",
    locator="""You are the **Locator Agent** in a multi-agent bug-fixing system.
Your sole job is DECOUPLING: break down a GitHub issue into independent, actionable sub-problems.

Rules:
1. Each sub-problem must be a SINGLE concern (one file, one function, one logic error).
2. Label each sub-problem with: [file], [function], [type: logic|config|dependency|api|test]
3. Assign a preliminary priority (P0=blocks everything, P1=core fix, P2=peripheral).
4. Do NOT write code. Do NOT propose fixes. Only decompose.

Output format:
SUB-1 [P0] file: xxx.py, func: yyy, type: logic
  What: <concise description>

SUB-2 [P1] file: aaa.py, func: bbb, type: api
  What: <concise description>
...
""",

    fixer="""You are the **Fixer Agent** in a multi-agent bug-fixing system.
Your job is EVALUATION → AGGREGATION: assess sub-problems and generate a concrete patch.

Rules:
1. Review the Locator's sub-problems. Re-evaluate each: is it a real problem? What happens if we skip it?
2. Re-prioritize with explicit criteria:
   - Does this sub-problem directly cause the reported bug?
   - What is the blast radius of the fix?
   - Can this sub-problem be safely ignored (attention abandonment)?
3. For sub-problems you commit to fixing, write a unified diff patch.
4. Mark sub-problems you are INTENTIONALLY NOT fixing as [ABANDONED] with a reason.

Output format:
## Priority Re-assessment
SUB-1 [P0→P0] KEEP — directly causes reported bug
SUB-2 [P1→P1] KEEP — follows from SUB-1 fix
SUB-3 [P2→ABANDONED] — cosmetic, not related to reported behavior

## Diff Patch
```diff
<unified diff>
```
""",

    reviewer="""You are the **Reviewer Agent** in a multi-agent bug-fixing system.
Your job is AGGREGATION VERIFICATION: check whether the patch actually solves the reported issue.

Rules:
1. Read the original issue statement carefully.
2. Compare it against the Fixer's patch.
3. Check for:
   - Does the patch address ALL symptoms described in the issue?
   - Does the patch introduce obvious new problems?
   - Were any sub-problems incorrectly abandoned?
4. Verdict: PASS or REJECT with specific reasons.

Output format:
VERDICT: PASS|REJECT

## Issues Found
- <specific problem, or "none">

## Suggestions for Fixer (if REJECT)
- <concrete correction>
""",
)


# ============================================================
# Meta-Analyst Agent — self-reflection between phases
# ============================================================

SYSTEM_META = """You are a **Meta-Analyst Agent** responsible for improving a multi-agent bug-fixing system.

You will be shown a batch of trajectory summaries from the system's runs. Each trajectory contains:
- The original GitHub issue
- What the Locator produced (sub-problems)
- What the Fixer produced (priority + patch)
- What the Reviewer decided (PASS/REJECT)
- Revision count

Your job: identify SYSTEMATIC PATTERNS of failure across the batch. Focus on:

**D (Decoupling) patterns**:
- Are issues being properly decomposed, or treated as monoliths?
- Are sub-problems at the right granularity?

**E (Evaluation) patterns**:
- Is the Fixer using explicit criteria for priority, or just guessing?
- Are there sub-problems that should have been abandoned but weren't?
- Is the Fixer justifying its priority decisions with reasoning?

**A (Aggregation) patterns**:
- Is the Reviewer catching real problems, or rubber-stamping?
- Are rejection→revision cycles actually improving patches?
- Are certain types of issues systematically missed?

Output format:
```
## Systematic Weaknesses Identified

### D (Decoupling)
- <pattern 1>
- <pattern 2>

### E (Evaluation)
- <pattern 1>
- <pattern 2>

### A (Aggregation)
- <pattern 1>
- <pattern 2>

## Concrete Prompt Improvements

### Locator Prompt Changes
<specific additions/modifications to the Locator's system prompt>

### Fixer Prompt Changes
<specific additions/modifications to the Fixer's system prompt>

### Reviewer Prompt Changes
<specific additions/modifications to the Reviewer's system prompt>
```
"""


# ============================================================
# Orchestrator — 3-phase
# ============================================================

class DEAPipeline:
    def __init__(self, llm: DeepSeekLLM, prompts: AgentPrompts, kb: KnowledgeBase,
                 max_revisions: int = 2):
        self.llm = llm
        self.prompts = prompts
        self.kb = kb
        self.max_revisions = max_revisions

    def run_one(self, instance: dict) -> dict:
        """Run the 3-agent pipeline on a single instance, with KB context."""
        inst_id = instance["instance_id"]
        repo = instance["repo"]
        difficulty = instance.get("difficulty", "unknown")
        problem = instance["problem_statement"]

        kb_ctx = self.kb.get_kb_context(max_entries=5)
        kb_summary = self.kb.get_pattern_summary()
        print(f"  [{inst_id}] ({kb_summary}) D...", end="", flush=True)

        # Step 1: Decoupling — KB context informs decomposition
        d_user = (f"{kb_ctx}\n\n"
                  f"## Current Issue\n{problem}")
        d_response = self.llm.chat(self.prompts.locator, d_user)
        subproblems = self._parse_subproblems(d_response)
        self.kb.add("Locator", inst_id, "subproblems", json.dumps(subproblems, ensure_ascii=False))
        print(f" {len(subproblems)} subs → E...", end="", flush=True)

        # Step 2: Evaluation → Fix — KB context + locator output
        e_user = (f"{kb_ctx}\n\n"
                  f"## Current Issue\n{problem}\n\n"
                  f"## Locator Output\n{d_response}")
        e_responses = [self.llm.chat(self.prompts.fixer, e_user)]
        current_patch = self._extract_patch(e_responses[0])

        # Step 3: Aggregation Review
        revision = 0
        final_verdict = "REJECT"
        a_response = ""

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

        # Heuristic flags
        flags = self._flag(subproblems, e_responses, final_verdict, current_patch)

        # Update KB with this instance's learnings
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
            }
        }

    def _flag(self, subproblems: list, e_responses: list, verdict: str, patch: str) -> list[str]:
        flags = []
        if len(subproblems) <= 1:
            flags.append("D_MONOLITH")
        elif len(subproblems) >= 8:
            flags.append("D_OVERFRAGMENT")
        for s in subproblems:
            if s.count("file:") >= 3:
                flags.append("D_BUNDLED")

        all_e = " ".join(e_responses)
        if "[ABANDONED]" not in all_e and len(subproblems) >= 3:
            flags.append("E_NO_ABANDON")
        if "P0" not in all_e and len(subproblems) >= 2:
            flags.append("E_FLAT_PRIORITY")
        if not re.search(r"(because|reason|due to|blast radius)", all_e, re.IGNORECASE):
            flags.append("E_NO_JUSTIFICATION")

        if verdict == "REJECT":
            flags.append("A_MAX_REVISIONS")
        if verdict == "PASS" and len(patch) < 50:
            flags.append("A_FALSE_PASS")
        return flags

    @staticmethod
    def _parse_subproblems(text: str) -> list[str]:
        subs = re.findall(r"SUB-\d+\s*\[P\d\].*?(?=SUB-\d+|$)", text, re.DOTALL)
        return [s.strip()[:300] for s in subs]

    @staticmethod
    def _extract_patch(text: str) -> str:
        m = re.search(r"```diff(.*?)```", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _parse_verdict(text: str) -> str:
        return "PASS" if re.search(r"VERDICT:\s*PASS", text, re.IGNORECASE) else "REJECT"


# ============================================================
# Phase Runner
# ============================================================

def run_phase(llm: DeepSeekLLM, prompts: AgentPrompts, instances: list,
              phase_name: str, out_dir: Path, kb: KnowledgeBase = None,
              start_idx: int = 0) -> tuple[list[dict], KnowledgeBase]:
    """Run one phase on all instances, save trajectories, return results + KB."""
    if kb is None:
        kb = KnowledgeBase(persistent_file=str(out_dir / f"{phase_name}_kb.json"))

    pipeline = DEAPipeline(llm, prompts, kb)
    phase_dir = out_dir / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, inst in enumerate(instances):
        inst_id = inst["instance_id"]
        print(f"\n[{phase_name}] {i+1}/{len(instances)}", end="")
        try:
            result = pipeline.run_one(inst)
            result["phase"] = phase_name
            result["prompt_version"] = prompts.version
            results.append(result)

            # Save trajectory
            traj_file = phase_dir / f"{inst_id.replace('/', '_')}.json"
            with open(traj_file, "w", encoding="utf-8") as f:
                json.dump(result["trajectory"], f, ensure_ascii=False, indent=2)

            # Save KB snapshot incrementally
            kb.save(str(out_dir / f"{phase_name}_kb.json"))
        except Exception as e:
            print(f"  !! Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "instance_id": inst_id, "phase": phase_name, "error": str(e),
                "flags": ["EXEC_ERROR"], "reviewer_verdict": "ERROR"
            })

    return results, kb


def self_reflection(llm: DeepSeekLLM, phase_results: list[dict],
                    kb: KnowledgeBase = None) -> AgentPrompts:
    """Meta-Analyst absorbs Phase 1 trajectories + KB, proposes prompt improvements."""
    print(f"\n{'='*60}")
    print("SELF-REFLECTION: Meta-Analyst absorbing Phase 1 KB + trajectories...")
    print(f"{'='*60}")

    # KB pattern summary
    kb_text = kb.get_pattern_summary() if kb else "No KB available."

    # Build summary of Phase 1 patterns
    summaries = []
    for r in phase_results:
        if r.get("error"):
            continue
        summaries.append({
            "instance_id": r["instance_id"],
            "repo": r.get("repo", ""),
            "num_subproblems": r.get("num_subproblems", 0),
            "flags": r.get("flags", []),
            "verdict": r.get("reviewer_verdict", ""),
            "revisions": r.get("revision_count", 0),
        })

    # Count flag patterns
    all_flags = [f for r in phase_results for f in r.get("flags", [])]
    d_count = sum(1 for f in all_flags if f.startswith("D_"))
    e_count = sum(1 for f in all_flags if f.startswith("E_"))
    a_count = sum(1 for f in all_flags if f.startswith("A_"))
    pass_rate = sum(1 for r in phase_results if r.get("reviewer_verdict") == "PASS")
    total = len(phase_results)

    # Get top trajectories by flag count for deep analysis
    top_flagged = sorted(
        [(r.get("instance_id"), len(r.get("flags", [])), r.get("flags", []))
         for r in phase_results if r.get("flags")],
        key=lambda x: x[1], reverse=True
    )[:5]

    reflection_prompt = f"""Phase 1 completed: {total} instances.

## KB Aggregate
{kb_text}

## Aggregate Statistics
- Reviewer PASS rate: {pass_rate}/{total} ({100*pass_rate/max(total,1):.0f}%)
- Flag distribution: D={d_count} | E={e_count} | A={a_count}
- E/(D+A) ratio: {e_count/max(d_count+a_count,1):.2f}
- Instances with 3+ flags: {sum(1 for r in phase_results if len(r.get('flags',[])) >= 3)}

## Most-flagged instances (worst D-E-A defects)
{json.dumps(top_flagged, ensure_ascii=False, indent=2)}

## Per-instance summary
{json.dumps(summaries, ensure_ascii=False, indent=2)}

Analyze systematic weaknesses and propose concrete prompt improvements.
"""

    meta_response = llm.chat(SYSTEM_META, reflection_prompt, temperature=0.4)

    # Parse proposed changes
    locator_changes = _extract_section(meta_response, "Locator Prompt Changes")
    fixer_changes = _extract_section(meta_response, "Fixer Prompt Changes")
    reviewer_changes = _extract_section(meta_response, "Reviewer Prompt Changes")

    # Build V2 prompts: original + self-reflection improvements
    v2 = AgentPrompts(
        version="v2.0_self_reflection",
        locator=PROMPTS_V1.locator + f"\n\n## Self-Reflection Improvements (from Phase 1 analysis)\n{locator_changes}" if locator_changes else PROMPTS_V1.locator,
        fixer=PROMPTS_V1.fixer + f"\n\n## Self-Reflection Improvements (from Phase 1 analysis)\n{fixer_changes}" if fixer_changes else PROMPTS_V1.fixer,
        reviewer=PROMPTS_V1.reviewer + f"\n\n## Self-Reflection Improvements (from Phase 1 analysis)\n{reviewer_changes}" if reviewer_changes else PROMPTS_V1.reviewer,
    )

    # Save reflection output
    with open(out_dir / "self_reflection_output.md", "w", encoding="utf-8") as f:
        f.write(f"# Self-Reflection Output\n\n{meta_response}\n\n")
        f.write(f"## Updated Prompts (v2.0_self_reflection)\n\n")
        f.write(f"### Locator\n```\n{v2.locator}\n```\n\n")
        f.write(f"### Fixer\n```\n{v2.fixer}\n```\n\n")
        f.write(f"### Reviewer\n```\n{v2.reviewer}\n```\n")

    print(f"\n  Self-reflection complete.")
    print(f"  Locator changes: {len(locator_changes)} chars")
    print(f"  Fixer changes: {len(fixer_changes)} chars")
    print(f"  Reviewer changes: {len(reviewer_changes)} chars")
    print(f"  → Saved to self_reflection_output.md")

    return v2


def _extract_section(text: str, heading: str) -> str:
    """Extract content under a markdown heading."""
    pattern = rf"###\s*{heading}\s*\n(.*?)(?=\n###|\n##|\Z)"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ""


def build_phase3_prompts(phase2_prompts: AgentPrompts, mimo_feedback_file: str) -> AgentPrompts:
    """Apply Mimo's external diagnosis feedback to create Phase 3 prompts.

    mimo_feedback_file should contain Mimo's D-E-A diagnosis and concrete prompt fix
    suggestions, copied from Open Code. Format is free-text; the function embeds
    the entire feedback as an appendix to each agent prompt.
    """
    try:
        with open(mimo_feedback_file, "r", encoding="utf-8") as f:
            feedback = f.read()
    except FileNotFoundError:
        print(f"  WARNING: Mimo feedback file '{mimo_feedback_file}' not found.")
        print(f"  Using Phase 2 prompts unchanged. Create the file and re-run.")
        feedback = "Mimo feedback pending — to be added after Open Code diagnosis."

    appendix = f"\n\n## External Diagnosis (Mimo 2.5 Pro — D-E-A Process Analysis)\n{feedback}"

    return AgentPrompts(
        version="v3.0_mimo_intervention",
        locator=phase2_prompts.locator + appendix,
        fixer=phase2_prompts.fixer + appendix,
        reviewer=phase2_prompts.reviewer + appendix,
    )


# ============================================================
# Phase comparison
# ============================================================

def compare_phases(phase_results: dict[str, list[dict]], out_dir: Path):
    """Generate cross-phase comparison table."""
    stats = {}
    for phase_name, results in phase_results.items():
        valid = [r for r in results if not r.get("error")]
        total = len(results)
        passes = sum(1 for r in valid if r.get("reviewer_verdict") == "PASS")
        all_flags = [f for r in valid for f in r.get("flags", [])]
        d_flags = sum(1 for f in all_flags if f.startswith("D_"))
        e_flags = sum(1 for f in all_flags if f.startswith("E_"))
        a_flags = sum(1 for f in all_flags if f.startswith("A_"))
        avg_subs = sum(r.get("num_subproblems", 0) for r in valid) / max(len(valid), 1)
        avg_revs = sum(r.get("revision_count", 0) for r in valid) / max(len(valid), 1)

        stats[phase_name] = {
            "total": total,
            "errors": total - len(valid),
            "pass_rate": f"{passes}/{len(valid)} ({100*passes/max(len(valid),1):.0f}%)",
            "avg_subproblems": round(avg_subs, 1),
            "avg_revisions": round(avg_revs, 1),
            "flag_distribution": f"D={d_flags} | E={e_flags} | A={a_flags}",
            "E_ratio": f"{e_flags/max(d_flags+a_flags, 1):.2f}",
        }

    comparison = {
        "phases": stats,
        "deltas": {
            "P1→P2_pass_rate_change": _delta_str(stats, "phase1", "phase2", "pass_rate"),
            "P2→P3_pass_rate_change": _delta_str(stats, "phase2", "phase3", "pass_rate"),
            "P1→P3_total_improvement": _delta_str(stats, "phase1", "phase3", "pass_rate"),
        },
        "interpretation": {
            "self_reflection_effect": (
                "Phase 2 improved over Phase 1 → agents can partially self-diagnose"
                if _is_improvement(stats, "phase1", "phase2")
                else "Phase 2 did NOT improve → D-E-A defects are invisible to the system itself"
            ),
            "external_diagnosis_effect": (
                "Phase 3 improved over Phase 2 → external diagnosis (Mimo) adds value beyond self-reflection"
                if _is_improvement(stats, "phase2", "phase3")
                else "Phase 3 did NOT improve → either self-reflection already captured gains, or external feedback wasn't actionable"
            ),
        }
    }

    comp_path = out_dir / "phase_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # Print comparison table
    print(f"\n{'='*70}")
    print("PHASE COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Phase 1 (Baseline)':<20} {'Phase 2 (Self)':<20} {'Phase 3 (Mimo)':<20}")
    print(f"{'-'*25} {'-'*20} {'-'*20} {'-'*20}")
    for metric in ["total", "pass_rate", "avg_subproblems", "avg_revisions", "flag_distribution", "E_ratio"]:
        p1 = stats.get("phase1", {}).get(metric, "N/A")
        p2 = stats.get("phase2", {}).get(metric, "N/A")
        p3 = stats.get("phase3", {}).get(metric, "N/A")
        print(f"{metric:<25} {str(p1):<20} {str(p2):<20} {str(p3):<20}")

    print(f"\n  Self-reflection effect: {comparison['interpretation']['self_reflection_effect']}")
    print(f"  External diagnosis effect: {comparison['interpretation']['external_diagnosis_effect']}")
    print(f"  Full comparison → {comp_path}")

    return comparison


def _delta_str(stats: dict, p1: str, p2: str, metric: str) -> str:
    """Compute string delta between two phases."""
    try:
        v1_str = stats.get(p1, {}).get(metric, "0/0 (0%)")
        v2_str = stats.get(p2, {}).get(metric, "0/0 (0%)")
        # Extract percentage
        m1 = re.search(r'(\d+)%', str(v1_str))
        m2 = re.search(r'(\d+)%', str(v2_str))
        if m1 and m2:
            delta = int(m2.group(1)) - int(m1.group(1))
            return f"{'+' if delta > 0 else ''}{delta}pp"
    except:
        pass
    return "N/A"


def _is_improvement(stats: dict, p1: str, p2: str) -> bool:
    try:
        v1 = stats.get(p1, {}).get("pass_rate", "0%")
        v2 = stats.get(p2, {}).get("pass_rate", "0%")
        m1 = re.search(r'(\d+)%', str(v1))
        m2 = re.search(r'(\d+)%', str(v2))
        return m1 and m2 and int(m2.group(1)) > int(m1.group(1))
    except:
        return False


# ============================================================
# Main — 3-phase run
# ============================================================

def main():
    from datasets import load_from_disk

    print("Loading SWE-bench Verified...")
    ds = load_from_disk("D:/数据/论文数据/SWE-bench_Verified")
    all_instances = list(ds)

    N = min(36, len(all_instances))
    instances = all_instances[:N]
    print(f"Using {N} instances (same set across all phases for within-instance comparison)\n")

    global out_dir
    out_dir = Path("D:/数据/论文数据/SWE-bench_DEA")
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = DeepSeekLLM()
    phase_results = {}

    # ================================================================
    # PHASE 1: Baseline (V1 prompts, KB starts empty)
    # ================================================================
    print(f"\n{'#'*70}")
    print(f"# PHASE 1: BASELINE (v1.0 prompts, KB from scratch)")
    print(f"{'#'*70}")
    results_p1, kb = run_phase(llm, PROMPTS_V1, instances, "phase1_baseline", out_dir)
    phase_results["phase1"] = results_p1

    with open(out_dir / "phase1_summary.json", "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "trajectory"} for r in results_p1],
                  f, ensure_ascii=False, indent=2)

    # Phase 1 KB pattern summary
    print(f"\n  Phase 1 KB: {kb.get_pattern_summary()}")
    print(f"  KB entries: {len(kb.entries)}")

    # ================================================================
    # SELF-REFLECTION: Meta-Analyst absorbs Phase 1 KB + trajectories
    # ================================================================
    print(f"\n{'#'*70}")
    print(f"# SELF-REFLECTION: Meta-Analyst absorbing Phase 1 patterns")
    print(f"{'#'*70}")

    prompts_v2 = self_reflection(llm, results_p1, kb)

    # ================================================================
    # PHASE 2: Self-reflection run (V2 prompts, fresh KB seeded from Phase 1 patterns)
    # ================================================================
    print(f"\n{'#'*70}")
    print(f"# PHASE 2: SELF-REFLECTION RUN (v2.0 prompts, KB from Phase 1)")
    print(f"{'#'*70}")
    results_p2, kb2 = run_phase(llm, prompts_v2, instances, "phase2_self_reflection", out_dir, kb)
    phase_results["phase2"] = results_p2

    with open(out_dir / "phase2_summary.json", "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "trajectory"} for r in results_p2],
                  f, ensure_ascii=False, indent=2)

    print(f"\n  Phase 2 KB: {kb2.get_pattern_summary()}")

    # Save prompts for Mimo to review
    prompts_v2_file = out_dir / "prompts_v2_for_mimo_review.json"
    with open(prompts_v2_file, "w", encoding="utf-8") as f:
        json.dump(prompts_v2.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\n  Phase 2 prompts saved → {prompts_v2_file}")
    print(f"  → Take this file + phase2 trajectories to Mimo in Open Code for external diagnosis.")
    print(f"  → After Mimo diagnosis, save feedback to: {out_dir / 'mimo_feedback.md'}")
    print(f"  → Then re-run with --phase3 to apply Mimo's feedback.\n")

    # ================================================================
    # PHASE 3: Mimo external diagnosis (requires mimo_feedback.md)
    # ================================================================
    mimo_feedback = out_dir / "mimo_feedback.md"
    if mimo_feedback.exists():
        print(f"\n{'#'*70}")
        print(f"# PHASE 3: MIMO EXTERNAL DIAGNOSIS")
        print(f"{'#'*70}")
        prompts_v3 = build_phase3_prompts(prompts_v2, str(mimo_feedback))
        results_p3, kb3 = run_phase(llm, prompts_v3, instances, "phase3_mimo_intervention", out_dir, kb2)
        phase_results["phase3"] = results_p3

        with open(out_dir / "phase3_summary.json", "w", encoding="utf-8") as f:
            json.dump([{k: v for k, v in r.items() if k != "trajectory"} for r in results_p3],
                      f, ensure_ascii=False, indent=2)
    else:
        print(f"\n{'#'*70}")
        print(f"# PHASE 3: SKIPPED (mimo_feedback.md not found)")
        print(f"#   → After Mimo diagnosis in Open Code, save feedback to:")
        print(f"#     {mimo_feedback}")
        print(f"#   → Then re-run: python agent_system.py --phase3-only")
        print(f"{'#'*70}")

    # ================================================================
    # Cross-phase comparison
    # ================================================================
    compare_phases(phase_results, out_dir)

    print(f"\n{'='*70}")
    print("ALL PHASES COMPLETE")
    print(f"  Phase 1 KB: {out_dir / 'phase1_baseline_kb.json'}")
    print(f"  Phase 2 KB: {out_dir / 'phase2_self_reflection_kb.json'}")
    print(f"  Phase 1 trajectories: {out_dir / 'phase1_baseline'}")
    print(f"  Phase 2 trajectories: {out_dir / 'phase2_self_reflection'}")
    print(f"  Self-reflection: {out_dir / 'self_reflection_output.md'}")
    if "phase3" in phase_results:
        print(f"  Phase 3 trajectories: {out_dir / 'phase3_mimo_intervention'}")
    print(f"  Comparison: {out_dir / 'phase_comparison.json'}")
    print(f"\n  Next step for you:")
    print(f"  1. Review self_reflection_output.md — what did the system identify on its own?")
    print(f"  2. Check KB files — did memory accumulate useful patterns?")
    print(f"  3. Compare Phase 1 vs Phase 2 pass rates — did self-reflection + KB help?")
    print(f"  4. Pick 5-10 flagged Phase 2 trajectories → Open Code → Mimo diagnosis")
    print(f"  5. Save Mimo's feedback to mimo_feedback.md → run --phase3-only")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys
    if "--phase3-only" in sys.argv:
        # Re-run only Phase 3 with existing Phase 2 prompts + Mimo feedback
        from datasets import load_from_disk
        out_dir = Path("D:/数据/论文数据/SWE-bench_DEA")
        ds = load_from_disk("D:/数据/论文数据/SWE-bench_Verified")
        instances = list(ds)[:36]
        llm = DeepSeekLLM()

        # Load Phase 2 prompts
        with open(out_dir / "prompts_v2_for_mimo_review.json", "r", encoding="utf-8") as f:
            p2_dict = json.load(f)
        prompts_v2 = AgentPrompts(**p2_dict)

        prompts_v3 = build_phase3_prompts(prompts_v2, str(out_dir / "mimo_feedback.md"))
        phase_results = {"phase3": run_phase(llm, prompts_v3, instances, "phase3_mimo_intervention", out_dir)}

        # Load previous phase summaries for comparison
        import glob
        phase_results["phase1"] = []
        phase_results["phase2"] = []
        # ... simplified: just run compare on phase3 alone if others missing
        print("Phase 3 complete. Run full pipeline for cross-phase comparison.")
    else:
        main()
