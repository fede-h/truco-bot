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

## 5. Model Hierarchy & Subagent Tiering

To maximize reasoning quality while protecting tokens, quotas, and context windows:

- **Orchestrator / Architect (Parent Session)**:
  - Powered by primary reasoning model (Claude / Gemini high tier).
  - Handles user interaction, grilling (`/grill-me`), spec creation (`/sdd-spec`), task graph orchestration (`/to-tickets`), and final synthesis.
  - Never clutters its context with repetitive compiler/test output or file exploration dumps.

- **Workers / Implementers (Child Subagents)**:
  - Dispatched for focused, bounded leaf tasks: exploration, code generation, single-ticket TDD, and invariant checks.
  - **Routing Ladder**:
    1. **Tier 1 (Free / Fast Models)**: Defaults to OpenCode free tiers (e.g., `opencode/deepseek-v4.1-flash-free`, `opencode/glm-5.3-flash-free`, `opencode/nemotron-3.5-lightning-free`, `opencode-go/beam-free`).
    2. **Tier 2 (Fallback / Engine Deep Code)**: If Tier 1 is unavailable or throttled, falls back to `google/gemini-3.8-flash` or `antigravity/antigravity-gemini-3.8-flash`.
    3. **Escalation**: If a worker fails after 2 iterations or hits deep conceptual roadblocks, control escalates to the Orchestrator.

