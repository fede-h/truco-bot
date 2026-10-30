# ADD Framework: User Guide

A practical guide on how to work with AI agents in this repository using the Agent-Driven Development (ADD) framework.

---

## Quick Reference: Slash Commands & Skills

| Task | Skill / Command | When to use |
|---|---|---|
| **Clarify an idea / feature** | `/grilling` or `/grill-me` | You have a feature or experiment idea and want an agent to stress-test your design before writing code. |
| **Triage issues & PRs** | `/triage` | Categorize incoming issues, mark missing info, or tag `ready-for-agent`. |
| **Write specification & ADRs** | `/sdd-spec` or `/grill-with-docs` | Define requirements, inputs, outputs, and capture decisions in `docs/adr/`. |
| **Decompose into tracer tickets** | `/to-tickets` | Break a spec down into manageable GitHub issues on Project #1 with dependencies. |
| **Implement with TDD** | `/tdd` or `/sdd-apply` | Direct the agent to write failing tests first (`cargo test` or `pytest`), then implement. |
| **Adversarial code review** | `/judgment-day` | Run a blind dual review (Spec Conformance + Engine Standards/Invariants) before merging. |
| **Audit for over-engineering** | `/ponytail-review` | Inspect code to eliminate unnecessary abstractions, boilerplate, and slop. |

---

## Core Workflows

### 1. Research & Ideation Flow (Human ➔ Agent)
When exploring a new CFR technique, engine optimization, or game rule:

1. **Ask for a design challenge**:
   ```
   /grill-me I want to add a neural network value head to prune subtrees during MCCFR traversal.
   ```
2. **Capture architectural decisions**:
   - The agent writes durable decisions to `docs/adr/XXXX-<title>.md`.
   - Any new terminology is added to `GLOSSARY.md`.

### 2. Feature Specification & Ticketing Flow
Once a concept is clear:

1. **Create the Spec**:
   ```
   /to-spec
   ```
   or invoke `/sdd-spec` to produce a delta spec.
2. **Break into Tickets**:
   ```
   /to-tickets
   ```
   This creates GitHub issues linked to [GitHub Project #1](https://github.com/users/fede-h/projects/1) in the **Ready** column.

### 3. Implementation Flow (Agent TDD)
When implementing a ticket:

1. Point the agent at the issue:
   ```
   /tdd Implement issue #5
   ```
2. **The Red-Green-Refactor invariant**:
   - The agent first writes a failing test in:
     - `crates/truco_engine/tests/` (Rust engine)
     - `tests/test_*.py` (Python wrappers / training scripts)
   - The agent runs tests (`cargo test` or `pytest tests/`) to observe failure.
   - The agent writes the minimal fix to pass.
   - Code is kept lean (Ponytail principle: standard library & minimal code first).

### 4. Review & Merge Flow (Adversarial Gate)
Before marking an issue **Done** or merging:

1. Run an adversarial dual-review:
   ```
   /judgment-day
   ```
2. Two independent reviews run in parallel:
   - **Spec Auditor**: Verifies whether the acceptance criteria were actually satisfied.
   - **Invariant Auditor**: Verifies no regressions in game rules, exploitability bounds, or FFI memory safety.
3. Once passed, close the issue and update the GitHub Project status to **Done**.

---

## Rules of Engagement

1. **Tests are non-negotiable**: Never accept a diff with non-trivial logic unless an executable test (`cargo test` / `pytest`) is added or updated.
2. **Minimal abstractions (Ponytail)**: Question speculative helpers, one-off classes, or config tables that aren't immediately used.
3. **Keep Docs in Sync**: If an agent introduces new domain vocabulary, ensure it records it in `GLOSSARY.md`.
