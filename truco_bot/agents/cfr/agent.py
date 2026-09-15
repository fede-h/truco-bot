import pickle
import random
import sqlite3
from collections.abc import Iterator, Mapping
from pathlib import Path

from truco_bot.agents.base import Agent
from truco_bot.core.actions import Action
from truco_bot.core.state import GameState, get_legal_actions, infoset_key
from truco_bot.env.obs import ID_TO_CARD


class SQLitePolicy(Mapping):
    """Lightweight zero-RAM read-only mapping backed by an SQLite B-tree policy table."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)

    def __getitem__(self, key: tuple) -> dict[Action, float]:
        k_bytes = pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL)
        row = self._conn.execute("SELECT a FROM policy WHERE k = ?", (k_bytes,)).fetchone()
        if row is None:
            raise KeyError(key)
        return pickle.loads(row[0])

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, tuple):
            return False
        k_bytes = pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL)
        row = self._conn.execute("SELECT 1 FROM policy WHERE k = ?", (k_bytes,)).fetchone()
        return row is not None

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM policy").fetchone()[0]

    def __iter__(self) -> Iterator[tuple]:
        for row in self._conn.execute("SELECT k FROM policy"):
            yield pickle.loads(row[0])

    def get(
        self, key: tuple, default: dict[Action, float] | None = None
    ) -> dict[Action, float] | None:
        k_bytes = pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL)
        row = self._conn.execute("SELECT a FROM policy WHERE k = ?", (k_bytes,)).fetchone()
        if row is None:
            return default
        return pickle.loads(row[0])

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


class VanillaCFRAgent(Agent):
    """Agent that chooses actions based on a CFR pre-trained policy table."""

    def __init__(
        self,
        policy: Mapping[tuple, dict[Action, float]] | None = None,
        is_canonical: bool = False,
        seed: int | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        self.policy: Mapping[tuple, dict[Action, float]] = policy if policy is not None else {}
        self.is_canonical: bool = is_canonical
        self.rng = random.Random(seed)
        self.db_path = str(db_path) if db_path else None
        self._conn: sqlite3.Connection | None = None
        if self.db_path:
            self._conn = sqlite3.connect(self.db_path)

        # Build secondary index for fast fallback lookup: (hand_key, current_trick, truco_level, pending_e, pending_t)
        self._fallback_index: dict[tuple, dict[Action, float]] = {}
        if not self.db_path and len(self.policy) <= 500_000:
            self._rebuild_fallback_index()

    def __del__(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def _rebuild_fallback_index(self) -> None:
        """Index policy states by coarse features for off-tree obs fallback."""
        for key, strat in self.policy.items():
            if len(key) >= 16:
                hand_key = key[0]
                tc = key[3]
                trick_idx = len(key[2])
                truco_lvl = key[10]
                pending_e = key[13]
                pending_t = key[15]
                feature = (hand_key, trick_idx, len(tc), truco_lvl, pending_e, pending_t)
                if feature not in self._fallback_index:
                    self._fallback_index[feature] = strat

    def save(self, filepath: str | Path) -> None:
        """Serialize and save the agent policy to a pickle checkpoint."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.policy, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_checkpoint(
        cls,
        filepath: str | Path,
        is_canonical: bool = False,
        seed: int | None = None,
    ) -> "VanillaCFRAgent":
        """Load an agent policy from a checkpoint file (pickle or sqlite)."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        if path.suffix in (".db", ".sqlite"):
            policy = SQLitePolicy(path)
            return cls(policy=policy, is_canonical=is_canonical, seed=seed, db_path=path)

        with open(path, "rb") as f:
            policy = pickle.load(f)
        db_candidate = path.with_suffix(".db")
        if not db_candidate.is_file():
            db_candidate = path.with_suffix(".sqlite")
        db_path = db_candidate if db_candidate.is_file() else None
        return cls(policy=policy, is_canonical=is_canonical, seed=seed, db_path=db_path)

    def _query_policy(self, key: tuple) -> dict[Action, float] | None:
        if self._conn is not None:
            k_bytes = pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL)
            row = self._conn.execute("SELECT a FROM policy WHERE k = ?", (k_bytes,)).fetchone()
            return pickle.loads(row[0]) if row else None
        return self.policy.get(key)

    def act_from_state(self, state: GameState) -> Action:
        """Choose an action given the exact GameState."""
        legal_actions = get_legal_actions(state)
        if not legal_actions:
            return Action.IR_AL_MAZO

        player = state.active_player
        if self.is_canonical:
            from truco_bot.agents.cfr.isomorphism import canonical_infoset_key

            key = canonical_infoset_key(state, player)
        else:
            key = infoset_key(state, player)

        action_probs = self._query_policy(key)
        if action_probs:
            return self._sample_action(action_probs, legal_actions)

        return self.rng.choice(legal_actions)

    def act(self, obs: tuple[int, ...], mask: list[bool]) -> Action:
        """Choose an action given the observation tuple and legal action mask."""
        legal_actions = [Action(i) for i, is_legal in enumerate(mask) if is_legal]
        if not legal_actions:
            return Action.IR_AL_MAZO
        if len(legal_actions) == 1:
            return legal_actions[0]

        hand_cards = [ID_TO_CARD[cid] for cid in obs[:3] if cid != 0]
        if self.is_canonical:
            from truco_bot.agents.cfr.isomorphism import canonicalize_hand

            hand_key = canonicalize_hand(hand_cards)
        else:
            hand_key = tuple(
                sorted(hand_cards, key=lambda c: (c.number, c.suit))
            )

        candidate_key = self._reconstruct_candidate_key(obs, mask, hand_key)
        action_probs = self._query_policy(candidate_key)
        if action_probs:
            return self._sample_action(action_probs, legal_actions)

        # Check secondary feature index
        pending_e = bool(mask[Action.QUIERO_ENVIDO.value]) if Action.QUIERO_ENVIDO.value < len(mask) else False
        pending_t = bool(mask[Action.QUIERO_TRUCO.value]) if Action.QUIERO_TRUCO.value < len(mask) else False
        tc_len = 1 if obs[3] != 0 else 0
        feature = (hand_key, obs[7], tc_len, obs[8], pending_e, pending_t)
        if feature in self._fallback_index:
            return self._sample_action(self._fallback_index[feature], legal_actions)

        # Off-tree uniform random fallback
        return self.rng.choice(legal_actions)

    def _reconstruct_candidate_key(
        self,
        obs: tuple[int, ...],
        mask: list[bool],
        hand_key: tuple,
    ) -> tuple:
        """Reconstruct candidate infoset key from observation vector."""
        active_player = obs[10]
        mano = obs[11]
        score_p0 = obs[5]
        score_p1 = obs[6]
        truco_level = obs[8]
        envido_resolved = bool(obs[9])

        pending_envido = (
            bool(mask[Action.QUIERO_ENVIDO.value])
            if Action.QUIERO_ENVIDO.value < len(mask)
            else False
        )
        pending_truco = (
            bool(mask[Action.QUIERO_TRUCO.value])
            if Action.QUIERO_TRUCO.value < len(mask)
            else False
        )

        tc: tuple = ()
        if obs[3] != 0:
            c0 = ID_TO_CARD[obs[3]]
            opp = 1 - active_player
            tc = ((opp, c0),)

        envido_winner = None
        envido_points = 0
        if envido_resolved:
            if score_p0 > 0:
                envido_winner = 0
                envido_points = score_p0
            elif score_p1 > 0:
                envido_winner = 1
                envido_points = score_p1

        return (
            hand_key,
            (),  # played_cards
            (),  # trick_results
            tc,
            mano,  # trick_leader
            mano,
            envido_winner,
            envido_points,
            (),  # envido_chain
            envido_resolved,
            truco_level,
            1 - active_player if pending_truco else None,
            1 - active_player if pending_envido else None,
            pending_envido,
            1 - active_player if pending_truco else None,
            pending_truco,
        )

    def _sample_action(
        self, 
        action_probs: dict[Action, float], 
        legal_actions: list[Action]
    ) -> Action:
        """Sample action from probability distribution over legal actions."""
        weights = [action_probs.get(a, 0.0) for a in legal_actions]
        if sum(weights) > 0.0:
            return self.rng.choices(legal_actions, weights=weights, k=1)[0]
        return self.rng.choice(legal_actions)


from truco_bot.agents.cfr.ensemble_agent import EnsembleCFRAgent
from truco_bot.agents.cfr.fallback import EquityFallback
from truco_bot.agents.cfr.history import HandHistoryTracker

__all__ = [
    "EnsembleCFRAgent",
    "EquityFallback",
    "HandHistoryTracker",
    "SQLitePolicy",
    "VanillaCFRAgent",
]

