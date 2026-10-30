# Design: Full Match (15/30 Points) Engine & WebSocket API

## Technical Approach
Implement a decoupled, deterministic multi-hand match engine (`TrucoMatch` and `TrucoMatchEnv`) on top of the base 1-hand engine. Fix base engine hand-level scoring and Envido window bugs first. Keep match scores strictly in `MatchState` to preserve trained CFR and Deep CFR models within their training distributions. Expose a single WebSocket route for interactive play with an optional simulated delay.

## Architecture Decisions

| Decision | Option Chosen | Alternatives Rejected | Rationale |
|---|---|---|---|
| **Engine Layer** | Python-first `MatchEngine` wrapping 1-hand `GameState` | Full Rust match rewrite | Match lifecycle has zero CPU bottleneck; Python gives immediate flexibility for API and agent dispatch. |
| **Score Memory** | Decoupled: `MatchState` stores running match score; `GameState` starts at (0,0) each hand | Passing running score to `GameState` | Subagent spike proved non-zero scores cause Deep CFR OOD degradation and corrupt Python `infoset_key`. |
| **Falta Envido** | Authentic Argentine rules: In *malas* (< half target), awards points to finish match ($target - winner$); in *buenas*, awards points leader needs ($target - leader$) | Symmetrical $target - \max(s0, s1)$ | Fixes rule asymmetry where trailing player couldn't win match on Falta in malas. |
| **Envido Timing** | Early match termination ("el envido va primero"); allow Pie to call Envido after Mano leads card | Requiring all tricks played; locking Envido on card 0 | Strictly aligns with official Argentine Truco rules. |
| **API Interface** | Single WebSocket endpoint (`/play`) | REST CRUD endpoints | Zero database or state CRUD bloat; real-time push events for bids and turn notifications. |
| **RL / Wrapper** | Lean `TrucoMatchEnv` mirroring `TrucoHandEnv` (0 external dependencies) | Full Gymnasium / PettingZoo adapters | Codebase uses no Gym dependencies; existing agents speak `act_from_state(state)` directly. |

## Data Flow

```
[WebSocket Client / Script]
             │
      JSON {type: action}
             ▼
      [MatchController]  ◄─── (Bot Delay Timer, Agent Registry)
             │
             ▼
        [TrucoMatch]
      ┌──────┴────────────────┐
      │ Match State:          │
      │ - match_score (p0, p1)│
      │ - mano rotation       │
      │ - target_score (15/30)│
      └──────┬────────────────┘
             │ instantiates (hands, mano, score=0,0)
             ▼
      [GameState (1 Hand)] ───► Hand Resolution ───► Earned Points (Envido + Truco)
                                                            │
                                  Match Score Updated ◄─────┘
```

## File Changes

| File | Action | Description |
|---|---|---|
| `truco_bot/core/rules.py` | Modify | Update `calculate_falta_envido_points` to support malas/buenas thresholding. |
| `truco_bot/core/state.py` | Modify | Allow Pie to sing Envido after Mano plays first card before trick 1 resolves. |
| `truco_bot/eval/showdown.py` | Modify | Fix hand delta to include `points_won` from Truco in addition to Envido. |
| `truco_bot/core/match.py` | Create | Pure state machine `MatchState` & `TrucoMatch`: mano rotation, early envido cutoff, 15/30 pts. |
| `truco_bot/env/match_env.py` | Create | Lean `TrucoMatchEnv` mirroring `TrucoHandEnv` interface for in-process bot/match runs. |
| `truco_bot/server/controller.py` | Create | `MatchSession` driving Human vs Bot, Bot vs Bot, and Human vs Human with delay toggle. |
| `truco_bot/server/app.py` | Create | Minimal FastAPI application exposing `ws://.../play`. |
| `tests/test_match.py` | Create | Unit tests for match scoring, early termination, mano rotation, and full 15/30 matches. |
| `tests/test_server.py` | Create | Tests for WebSocket session flow across all play modes. |

## Interfaces / Contracts

```python
@dataclass(slots=True)
class MatchState:
    target_score: int              # 15 or 30
    match_score_p0: int = 0
    match_score_p1: int = 0
    mano: int = 0                  # Alternates each hand
    hand_index: int = 0
    current_hand: GameState | None = None
    is_match_done: bool = False
    match_winner: int | None = None

class TrucoMatch:
    def __init__(self, target_score: int = 30, seed: int | None = None) -> None: ...
    def reset(self, seed: int | None = None) -> MatchState: ...
    def next_hand(self) -> GameState: ...
    def step(self, action: Action) -> tuple[MatchState, bool]: ...
    def get_view(self, player: int) -> dict[str, Any]: ...
```

## Testing Strategy

| Layer | What to Test | Approach |
|---|---|---|
| Unit | Authentic Falta Envido in malas vs buenas | Parametrized test asserting exact points awarded. |
| Unit | Mano first card $\to$ Pie envido sequence | Step Mano card 0, assert `Action.ENVIDO` in legal actions. |
| Unit | Early Envido match termination | Score 28-28 in 30 pt match, call Real Envido, assert `is_match_done == True` before tricks. |
| Integration | Full 15/30 point matches (Random, Heuristic, MCCFR) | Run 100 simulated matches, assert termination, monotonic scores, mano alternation. |
| E2E | WebSocket session protocol | Async test client connecting to `/play`, playing full match human vs bot. |

## Migration / Rollout
No migration required. All changes backward compatible with existing single-hand models and tests.

## Open Questions
None. All design questions settled in grilling rounds 1–5.
