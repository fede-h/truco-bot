# Truco Bot

A high-performance Argentine Truco (1v1) simulation engine, game-theoretic equilibrium solver (CFR / MCCFR+ / Deep CFR), and interactive WebSocket match server.

---

## Quickstart

### Prerequisites
- Python 3.12+ with [`uv`](https://docs.astral.sh/uv/)
- Rust 1.80+ with `cargo`

### Build Native Engine
```bash
uv sync
uv run maturin develop
```

---

## Running Matches & Benchmarks

### 15 / 30 Point Match Showdown
Play full matches between any combination of agents (`Random`, `Heuristic`, `Equity`, native CFR checkpoints, or Deep CFR):

```bash
# Heuristic vs Random (15-point match, 50 matches)
uv run python scripts/run_match_showdown.py --agent0 Heuristic --agent1 Random --matches 50 --target-score 15

# Native MCCFR+ checkpoint vs Heuristic (30-point match)
uv run python scripts/run_match_showdown.py --agent0 models/mccfr_plus_1000M.bin --agent1 Heuristic --matches 20 --target-score 30
```

---

## Interactive WebSocket Server

Run the FastAPI real-time gameplay server:

```bash
uv run uvicorn truco_bot.server.app:app --host 0.0.0.0 --port 8000
```

Connect to `ws://localhost:8000/play` with JSON frames:

```json
// Start a 15-point match vs MCCFR bot with simulated human thinking delay
{
  "type": "start",
  "mode": "human_vs_bot",
  "bot": "mccfr_1000m",
  "target_score": 15,
  "simulate_delay": true
}

// Play an action
{
  "type": "action",
  "action": "PLAY_CARD_0"
}
```

---

## Testing

```bash
# Run Rust bitboard engine tests
cargo test --manifest-path crates/truco_engine/Cargo.toml

# Run Python match & invariant tests
uv run pytest tests/test_match_engine.py tests/test_ticket_9_invariants.py tests/test_server.py
```
