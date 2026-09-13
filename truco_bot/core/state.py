from dataclasses import dataclass, field

from truco_bot.core.actions import Action
from truco_bot.core.card import Card
from truco_bot.core.rules import (
    calculate_envido,
    calculate_falta_envido_points,
    compare_cards,
    resolve_hand,
)


@dataclass(slots=True)
class GameState:
    hands: list[list[Card]]
    mano: int
    active_player: int
    score_p0: int = 0
    score_p1: int = 0
    max_score: int = 30
    current_trick: int = 0
    trick_cards: list[tuple[int, Card]] = field(default_factory=list)
    trick_results: list[int] = field(default_factory=list)
    played_cards: list[tuple[int, Card]] = field(default_factory=list)
    trick_leader: int = 0

    envido_resolved: bool = False
    envido_chain: list[Action] = field(default_factory=list)
    pending_envido_response: bool = False
    pending_envido_from: int | None = None

    truco_level: int = 1
    truco_caller: int | None = None
    pending_truco_response: bool = False
    pending_truco_from: int | None = None

    is_hand_done: bool = False
    winner: int | None = None
    points_won: int = 0


def create_initial_hand_state(
    hands: list[list[Card]],
    mano: int = 0,
    score_p0: int = 0,
    score_p1: int = 0,
    max_score: int = 30,
) -> GameState:
    return GameState(
        hands=hands,
        mano=mano,
        active_player=mano,
        score_p0=score_p0,
        score_p1=score_p1,
        max_score=max_score,
        trick_leader=mano,
    )


_EMPTY_ACTIONS: list[Action] = []
_ENVIDO_RESPONSES: list[Action] = [Action.QUIERO_ENVIDO, Action.NO_QUIERO_ENVIDO]
_ENVIDO_CALL_ENVIDO: list[Action] = [
    Action.QUIERO_ENVIDO,
    Action.NO_QUIERO_ENVIDO,
    Action.ENVIDO,
    Action.REAL_ENVIDO,
    Action.FALTA_ENVIDO,
]
_ENVIDO_CALL_REAL: list[Action] = [
    Action.QUIERO_ENVIDO,
    Action.NO_QUIERO_ENVIDO,
    Action.FALTA_ENVIDO,
]
_TRUCO_RESPONSES_1: list[Action] = [
    Action.QUIERO_TRUCO,
    Action.NO_QUIERO_TRUCO,
    Action.IR_AL_MAZO,
]
_TRUCO_RESPONSES_2: list[Action] = [
    Action.QUIERO_TRUCO,
    Action.NO_QUIERO_TRUCO,
    Action.IR_AL_MAZO,
    Action.VALE_CUATRO,
]
_TRUCO_RESPONSES_3: list[Action] = [
    Action.QUIERO_TRUCO,
    Action.NO_QUIERO_TRUCO,
    Action.IR_AL_MAZO,
]


def get_legal_actions(state: GameState) -> list[Action]:
    if state.is_hand_done:
        return _EMPTY_ACTIONS.copy()

    player = state.active_player

    if state.pending_envido_response:
        last_call = state.envido_chain[-1] if state.envido_chain else None
        if last_call == Action.ENVIDO:
            if state.envido_chain.count(Action.ENVIDO) >= 2:
                return [Action.QUIERO_ENVIDO, Action.NO_QUIERO_ENVIDO, Action.REAL_ENVIDO, Action.FALTA_ENVIDO]
            return _ENVIDO_CALL_ENVIDO.copy()
        elif last_call == Action.REAL_ENVIDO:
            return _ENVIDO_CALL_REAL.copy()
        return _ENVIDO_RESPONSES.copy()

    if state.pending_truco_response:
        if state.truco_level == 2:
            return _TRUCO_RESPONSES_2.copy()
        elif state.truco_level == 3:
            return _TRUCO_RESPONSES_3.copy()
        return _TRUCO_RESPONSES_1.copy()

    actions = []
    # envido
    if (
        state.current_trick == 0
        and not state.envido_resolved
        and len(state.trick_cards) < 2
        and len(state.envido_chain) == 0
    ):
        actions.extend([Action.ENVIDO, Action.REAL_ENVIDO, Action.FALTA_ENVIDO])

    # truco
    if state.truco_level == 1:
        actions.append(Action.TRUCO)
    elif state.truco_level == 2 and state.truco_caller != player:
        actions.append(Action.RETRUCO)
    elif state.truco_level == 3 and state.truco_caller != player:
        actions.append(Action.VALE_CUATRO)

    # card plays
    hand = state.hands[player]
    if len(hand) > 0:
        actions.append(Action.PLAY_CARD_0)
        actions.append(Action.PLAY_CARD_DOWN_0)
    if len(hand) > 1:
        actions.append(Action.PLAY_CARD_1)
        actions.append(Action.PLAY_CARD_DOWN_1)
    if len(hand) > 2:
        actions.append(Action.PLAY_CARD_2)
        actions.append(Action.PLAY_CARD_DOWN_2)

    actions.append(Action.IR_AL_MAZO)
    return actions


# ponytail: fast shallow clone replaces copy.deepcopy for 30x throughput
def _clone_state(state: GameState) -> GameState:
    hands_copy = (
        [state.hands[0].copy(), state.hands[1].copy()]
        if len(state.hands) == 2
        else [h.copy() for h in state.hands]
    )
    return GameState(
        hands_copy,
        state.mano,
        state.active_player,
        state.score_p0,
        state.score_p1,
        state.max_score,
        state.current_trick,
        state.trick_cards.copy(),
        state.trick_results.copy(),
        state.played_cards.copy(),
        state.trick_leader,
        state.envido_resolved,
        state.envido_chain.copy(),
        state.pending_envido_response,
        state.pending_envido_from,
        state.truco_level,
        state.truco_caller,
        state.pending_truco_response,
        state.pending_truco_from,
        state.is_hand_done,
        state.winner,
        state.points_won,
    )


def step(state: GameState, action: Action) -> GameState:
    new_state = _clone_state(state)
    player = new_state.active_player
    other_player = 1 - player

    if action == Action.IR_AL_MAZO:
        new_state.is_hand_done = True
        new_state.winner = other_player
        new_state.points_won = new_state.truco_level
        return new_state

    if action in [Action.ENVIDO, Action.REAL_ENVIDO, Action.FALTA_ENVIDO]:
        new_state.envido_chain.append(action)
        new_state.pending_envido_response = True
        new_state.pending_envido_from = other_player
        new_state.active_player = other_player
        return new_state

    if action == Action.QUIERO_ENVIDO:
        new_state.envido_chain.append(action)
        new_state.pending_envido_response = False
        new_state.envido_resolved = True
        new_state.active_player = (
            other_player if len(new_state.trick_cards) % 2 == 1 else new_state.trick_leader
        )

        envido_p0 = calculate_envido(new_state.hands[0])
        envido_p1 = calculate_envido(new_state.hands[1])
        winner = 0 if envido_p0 > envido_p1 else (1 if envido_p1 > envido_p0 else new_state.mano)

        if Action.FALTA_ENVIDO in new_state.envido_chain:
            pts = calculate_falta_envido_points(
                new_state.score_p0, new_state.score_p1, new_state.max_score
            )
        else:
            pts = 0
            for call in new_state.envido_chain:
                if call == Action.ENVIDO:
                    pts += 2
                elif call == Action.REAL_ENVIDO:
                    pts += 3
        if winner == 0:
            new_state.score_p0 = min(new_state.max_score, new_state.score_p0 + pts)
        else:
            new_state.score_p1 = min(new_state.max_score, new_state.score_p1 + pts)

        return new_state

    if action == Action.NO_QUIERO_ENVIDO:
        new_state.envido_chain.append(action)
        new_state.pending_envido_response = False
        new_state.envido_resolved = True
        new_state.active_player = (
            other_player if len(new_state.trick_cards) % 2 == 1 else new_state.trick_leader
        )

        caller = other_player
        chain = new_state.envido_chain
        if len(chain) == 2:
            pts = 1
        else:
            pts = 0
            for call in chain[:-2]:
                if call == Action.ENVIDO:
                    pts += 2
                elif call == Action.REAL_ENVIDO:
                    pts += 3
            if pts == 0:
                pts = 1

        if caller == 0:
            new_state.score_p0 += pts
        else:
            new_state.score_p1 += pts

        return new_state

    if action in [Action.TRUCO, Action.RETRUCO, Action.VALE_CUATRO]:
        new_state.pending_truco_response = True
        new_state.pending_truco_from = other_player
        new_state.truco_caller = player
        new_state.active_player = other_player
        if action == Action.VALE_CUATRO:
            new_state.truco_level = 3
        if not new_state.envido_resolved:
            new_state.envido_resolved = True
        return new_state

    if action == Action.QUIERO_TRUCO:
        new_state.pending_truco_response = False
        new_state.truco_level += 1
        new_state.active_player = (
            other_player if len(new_state.trick_cards) % 2 == 1 else new_state.trick_leader
        )
        return new_state

    if action == Action.NO_QUIERO_TRUCO:
        new_state.is_hand_done = True
        new_state.winner = other_player
        new_state.points_won = new_state.truco_level
        return new_state

    if 0 <= action.value <= 5:  # PLAY_CARD
        if not new_state.envido_resolved:
            new_state.envido_resolved = True

        card_idx = action.value % 3
        card = new_state.hands[player].pop(card_idx)
        new_state.trick_cards.append((player, card))
        new_state.played_cards.append((player, card))

        if len(new_state.trick_cards) == 2:
            p0_card = (
                new_state.trick_cards[0][1]
                if new_state.trick_cards[0][0] == 0
                else new_state.trick_cards[1][1]
            )
            p1_card = (
                new_state.trick_cards[1][1]
                if new_state.trick_cards[1][0] == 1
                else new_state.trick_cards[0][1]
            )

            res = compare_cards(p0_card, p1_card)
            new_state.trick_results.append(res)

            hr = resolve_hand(new_state.trick_results, new_state.mano)
            if hr is not None:
                new_state.is_hand_done = True
                new_state.winner = hr
                new_state.points_won = new_state.truco_level
            else:
                new_state.current_trick += 1
                if res != 0:
                    winner_player = 0 if res == 1 else 1
                    new_state.trick_leader = winner_player
                    new_state.active_player = winner_player
                else:
                    new_state.active_player = new_state.trick_leader
                new_state.trick_cards = []
        else:
            new_state.active_player = other_player

    return new_state


def infoset_key(state:GameState, player:int) -> tuple:
    """Flat, hashable tuple of everything a player can see."""
    envido_results:tuple[int | None, int] = (None, 0) # (envido winner, envido points)

    if state.envido_resolved:
        if state.score_p0 > 0:
            envido_results = (0, state.score_p0)
        elif state.score_p1 > 0:
            envido_results = (1, state.score_p1)
        else: envido_results = (None, 0)
    
    envido_winner = envido_results[0]
    envido_points = envido_results[1]

    infoset:tuple = (
        # the hand is sorted so same hands do not get repeated
        tuple(sorted(state.hands[player], key=lambda c: (c.number, c.suit))),
        tuple(state.played_cards),
        tuple(state.trick_results), tuple(state.trick_cards), state.trick_leader,
        state.mano,
        envido_winner, envido_points,
        tuple(state.envido_chain), state.envido_resolved,
        state.truco_level, state.truco_caller,
        state.pending_envido_from, state.pending_envido_response,
        state.pending_truco_from, state.pending_truco_response
    )

    return infoset