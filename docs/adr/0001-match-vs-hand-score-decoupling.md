# ADR-0001: Match vs Hand Score Decoupling & Match Engine Lifecycle

## Status
Accepted

## Context
Truco matches are played up to 15 or 30 points across multiple hands. Previously, the engine only supported isolated 1-hand episodes (`GameState` / `BitboardState`).

When designing the multi-hand match engine (`TrucoMatch`), we considered passing running match scores directly into `GameState.score_p0` and `score_p1`. However, an epistemic spike revealed critical breakages:
1. **Deep CFR Neural Policy Degradation**: Features 154/155 encode `score / max_score`. During training, the ResMLP policy was only exposed to single-hand values (`0.0` or small values $2..7$ after envido). Feeding running match scores ($0.4..1.0$) caused out-of-distribution (OOD) failure.
2. **Infoset Key Corruption**: In `core/state.py:334`, `infoset_key` checks `if state.score_p0 > 0`, falsely flagging that Player 0 won Envido with match points.
3. **History Tracking**: `HandHistoryTracker` uses score differences from $(0, 0)$ to track accepted/rejected Envido bids.

## Decision
1. **Score Decoupling**: Running match scores (`match_score_p0`, `match_score_p1`) are strictly isolated inside `MatchState`. Each hand's `GameState` is always initialized with `(0, 0)`. Earned points (`envido_pts + truco_pts`) are added to match scores upon hand resolution.
2. **Authentic Falta Envido**:
   - In *malas* ($< \text{target} // 2$): Awards points needed for the winner to finish the match ($\text{target} - \text{winner\_score}$).
   - In *buenas* ($\ge \text{target} // 2$): Awards points needed for the leader to finish ($\text{target} - \max(\text{scores})$).
3. **"El Envido Va Primero"**: If a player reaches the target score upon Envido resolution, the match transitions to `is_match_done = True` immediately without playing out trick cards.
4. **WebSocket Pacing Toggle**: `simulate_delay: bool` (default 0.5s for human vs bot, 0s for bots/benchmarks) ensures realistic pacing for interactive human play.

## Consequences
- **Positive**: 100% backward compatibility with all trained CFR checkpoints (`NativeCFRAgent`) and Deep CFR networks. No retrain required for 15/30-point match play.
- **Positive**: Clear separation between single-hand physics (Rust bitboard / PyO3) and multi-hand lifecycle (Python match state machine).
