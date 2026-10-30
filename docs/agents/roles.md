# Agent & Human Roles

Defines the roles and responsibilities in the Agent-Driven Development (ADD) lifecycle for `truco-bot`.

## 1. Researcher / Strategist (Human Driver)
- **Scope**: Game theory hypotheses, ML architectures (MCCFR, Deep CFR), tournament benchmarking, reward metrics, and high-level gameplay rules.
- **Tools / Artifacts**: `research/artifacts/`, benchmark runners (`scripts/`), experiment specs.
- **Boundary**: Drives the problem statement; avoids micro-managing implementation syntax.

## 2. Architect / Critic (Adversarial Design)
- **Scope**: System seams, Rust FFI boundaries, infoset encoding efficiency, zero-alloc state representations, and invariant verification.
- **Trigger Skills**: `/grill-me`, `/sdd-design`, `/domain-modeling`.
- **Outputs**: Architecture decisions (`docs/adr/`), domain glossary (`GLOSSARY.md`), and delta specs.

## 3. Implementer (TDD Builder)
- **Scope**: Red-Green-Refactor execution. Writes the minimal failing test first, then minimal working implementation.
- **Trigger Skills**: `/tdd`, `/sdd-apply`, `/implement`.
- **Target Areas**:
  - Rust Engine (`crates/truco_engine`): Fast bitboard/state transitions, isomorphism maps, MCCFR traversal (`cargo test`).
  - Python Package (`src/` / `scripts/`): Network trainers, FFI glue, evaluation harnesses (`pytest tests/`).

## 4. Adversary / Judge (Dual Reviewer)
- **Scope**: Independent verification before merge. Tests edge cases, invariant leaks, exploitability regressions, and spec compliance.
- **Trigger Skills**: `/judgment-day`, `/code-review`, `/ponytail-review`.
- **Rule**: Code does not close tickets or merge without passing both:
  1. Spec compliance (did it build what was asked?).
  2. Standards & Invariants (did it break CFR convergence, safety, or engine speed?).
