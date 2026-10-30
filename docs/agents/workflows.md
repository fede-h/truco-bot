# ADD Workflows: SDD, TDD & Issue Pipeline

Operating framework connecting GitHub Project #1, Spec-Driven Development, and Test-Driven Development.

## 1. Project Board Pipeline

GitHub Project #1 uses standard states mapped to agent activities:

| Project Status | Entry Trigger | Active Skills | Exit Criteria / Labels |
|---|---|---|---|
| **Backlog** | Issue created or researched | `/triage`, `/grilling` | Scope sharpened (`needs-triage` -> `ready-for-agent` / `ready-for-human`) |
| **Ready** | Specification validated | `/sdd-spec`, `/to-tickets` | DAG broken into clear sub-issues |
| **In progress** | Ticket claimed | `/tdd`, `/sdd-apply` | Red-Green-Refactor loop complete, tests pass |
| **In review** | Implementation ready | `/judgment-day`, `/code-review` | Dual review clean (Spec + Invariants) |
| **Done** | PR merged / tests green | `/work-unit-commits` | Issue closed with pointer comment |

## 2. SDD Protocol (Spec-Driven Development)

1. **Explore & Challenge**: Run `/sdd-explore` or `/grilling` on complex changes (engine mechanics, reward schemes).
2. **Spec & Design**:
   - Write concrete input/output delta specs via `/sdd-spec`.
   - Record durable architectural tradeoffs in `docs/adr/XXXX-<title>.md`.
   - Keep domain terms in `GLOSSARY.md`.
3. **Task Breakdown**: Use `/to-tickets` to generate tracer issues with clear dependency edges.

## 3. TDD Protocol (Test-Driven Development)

No feature or bugfix lands without an executable test proving the change:

- **Red Phase**: Write the smallest runnable test asserting the new invariant or reproducing the bug.
  - Rust: In `crates/truco_engine/tests/` or inline `#[cfg(test)]`.
  - Python: In `tests/test_<feature>.py`.
- **Green Phase**: Implement the minimal code that satisfies the test.
- **Refactor Phase**: Optimize for memory and speed (especially in Rust engine) without breaking test invariants.
- **Check Commands**:
  - `cargo test --workspace`
  - `pytest tests/`

## 4. Adversarial Merge Gate

Before any significant issue is closed or PR merged:
- Run `/judgment-day` to invoke independent reviewer subagents.
- Ensure no exploitability regressions or undocumented invariant changes occur.

## 5. Subagent Dispatch & Execution Rules

1. **AFK Criterion**: Dispatch to a subagent only if the task runs without interactive user steering (e.g., automated exploration, ticket TDD slice, standards review).
2. **Context Pointers Only**: Never dump parent chat logs into a subagent. Pass only pointers: ticket number/file, relevant interface, and repo path.
3. **Model Selection**:
   - Leaf tasks default to **Tier 1 Free Models** (`opencode/*-free`) via the `subagent` tool's `model` parameter.
   - Fall back to **Tier 2 Fast Models** (`gemini-3.8-flash`) if Tier 1 is unavailable or rate-limited.
4. **Worktree Isolation**: If multiple implementer subagents run concurrently, each runs in its own git worktree on an isolated branch to prevent index and HEAD collisions.
5. **No Subagent Recursion**: Subagents must execute their own tasks and never spawn recursive child subagents.

