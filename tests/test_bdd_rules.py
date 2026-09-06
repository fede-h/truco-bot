# ponytail: bdd steps for rules
from pytest_bdd import given, scenarios, then, when

from truco_bot.core.actions import Action
from truco_bot.core.card import Suit, create_card
from truco_bot.core.deck import deal
from truco_bot.core.rules import calculate_envido, resolve_hand
from truco_bot.core.state import create_initial_hand_state, step

scenarios("features/truco_rules.feature")


# -- Scenario 1 --
@given(
    "a new hand where Mano wins the second trick after a parda in the first trick",
    target_fixture="trick_results",
)
def given_parda_then_mano_win():
    # 0 is parda, 1 is Mano (P0) win
    return [0, 1]


@when("the hand is resolved", target_fixture="hand_winner")
def when_hand_resolved(trick_results):
    return resolve_hand(trick_results, mano=0)


@then("Mano wins the hand")
def then_mano_wins(hand_winner):
    assert hand_winner == 0


# -- Scenario 2 --
@given(
    "a player is dealt a hand with 10 of Copas, 11 of Copas, and 2 of Bastos",
    target_fixture="envido_hand",
)
def given_figure_hand():
    return [create_card(10, Suit.COPAS), create_card(11, Suit.COPAS), create_card(2, Suit.BASTOS)]


@when("the player calculates their envido", target_fixture="calculated_envido")
def when_calculate_envido(envido_hand):
    return calculate_envido(envido_hand)


@then("the envido score should be 20")
def then_envido_20(calculated_envido):
    assert calculated_envido == 20


# -- Scenario 3 --
@given("a game is dealt", target_fixture="game_state")
def given_game_dealt():
    hands, _ = deal(2, 3)
    return create_initial_hand_state(hands, mano=0)


@when("player 0 calls Truco", target_fixture="game_state")
def when_player_calls_truco(game_state):
    return step(game_state, Action.TRUCO)


@when("player 1 refuses Truco", target_fixture="game_state")
def when_player_refuses_truco(game_state):
    return step(game_state, Action.NO_QUIERO_TRUCO)


@then("player 0 wins the hand with 1 points")
def then_player_0_wins(game_state):
    assert game_state.is_hand_done
    assert game_state.winner == 0
    assert game_state.points_won == 1
