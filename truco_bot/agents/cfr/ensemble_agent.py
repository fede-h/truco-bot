"""Ensemble CFR Agent with history tracking and equity fallback."""

import pickle
from collections.abc import Mapping
from pathlib import Path

from truco_bot.agents.cfr.agent import SQLitePolicy, VanillaCFRAgent
from truco_bot.agents.cfr.fallback import EquityFallback
from truco_bot.agents.cfr.history import HandHistoryTracker
from truco_bot.core.actions import Action
from truco_bot.core.state import GameState, get_legal_actions, infoset_key
from truco_bot.env.obs import ID_TO_CARD


class EnsembleCFRAgent(VanillaCFRAgent):
    """Ensemble CFR agent combining pre-trained policy with stateful history and equity fallback."""

    __slots__ = ("fallback", "tracker")


    def __init__(
        self,
        policy: Mapping[tuple, dict[Action, float]] | None = None,
        is_canonical: bool = True,
        seed: int | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            policy=policy,
            is_canonical=is_canonical,
            seed=seed,
            db_path=db_path,
        )
        self.tracker = HandHistoryTracker()
        self.fallback = EquityFallback()

    def reset(self) -> None:
        """Reset internal history tracker for a new hand."""
        self.tracker.reset_hand()

    def act_from_state(self, state: GameState) -> Action:
        """Choose action given exact GameState, delegating off-tree states to EquityFallback."""
        legal_actions = get_legal_actions(state)
        if not legal_actions:
            return Action.IR_AL_MAZO
        if len(legal_actions) == 1:
            return legal_actions[0]

        player = state.active_player
        if self.is_canonical:
            from truco_bot.agents.cfr.isomorphism import canonical_infoset_key

            key = canonical_infoset_key(state, player)
        else:
            key = infoset_key(state, player)

        action_probs = self._query_policy(key)
        if action_probs:
            vals = list(action_probs.values())
            if len(vals) <= 1 or max(vals) - min(vals) >= 1e-6:
                return self._sample_action(action_probs, legal_actions)

        return self.fallback.decide_state(state, legal_actions)

    def act(self, obs: tuple[int, ...], mask: list[bool]) -> Action:
        """Choose action from observation and mask, consulting policy then EquityFallback."""
        self.tracker.update(obs, mask)

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
            vals = list(action_probs.values())
            if len(vals) <= 1 or max(vals) - min(vals) >= 1e-6:
                return self._sample_action(action_probs, legal_actions)

        return self.fallback.act(obs, mask)

    def _reconstruct_candidate_key(
        self,
        obs: tuple[int, ...],
        mask: list[bool],
        hand_key: tuple,
    ) -> tuple:
        """Reconstruct candidate infoset key with stateful history from tracker."""
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

        trick_leader = (
            self.tracker.last_trick_leader
            if self.tracker.played_cards
            else mano
        )

        return (
            hand_key,
            tuple(self.tracker.played_cards),
            tuple(self.tracker.trick_results),
            tc,
            trick_leader,
            mano,
            envido_winner,
            envido_points,
            tuple(self.tracker.envido_chain),
            envido_resolved,
            truco_level,
            1 - active_player if pending_truco else None,
            1 - active_player if pending_envido else None,
            pending_envido,
            1 - active_player if pending_truco else None,
            pending_truco,
        )

    @classmethod
    def from_checkpoint(
        cls,
        filepath: str | Path,
        is_canonical: bool = True,
        seed: int | None = None,
    ) -> "EnsembleCFRAgent":
        """Load an EnsembleCFRAgent from a checkpoint file (pickle or sqlite)."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        if path.suffix in (".db", ".sqlite"):
            policy = SQLitePolicy(path)
            return cls(
                policy=policy,
                is_canonical=is_canonical,
                seed=seed,
                db_path=path,
            )

        with open(path, "rb") as f:
            policy = pickle.load(f)
        db_candidate = path.with_suffix(".db")
        if not db_candidate.is_file():
            db_candidate = path.with_suffix(".sqlite")
        db_path = db_candidate if db_candidate.is_file() else None
        return cls(
            policy=policy,
            is_canonical=is_canonical,
            seed=seed,
            db_path=db_path,
        )


__all__ = ["EnsembleCFRAgent", "EquityFallback", "HandHistoryTracker"]
