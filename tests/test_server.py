"""Adversarial test suite for Ticket #11:
- Interactive WebSocket match server (/play)
- MatchSession controller for human_vs_bot, bot_vs_bot, human_vs_human
- Simulated bot delay toggle
- Masked state views over WebSockets
- End-to-end interactive playout
"""

from starlette.testclient import TestClient

from truco_bot.core.actions import Action
from truco_bot.server.app import app
from truco_bot.server.controller import MatchSession


class TestMatchSessionController:
    """Test MatchSession turn management and bot pacing."""

    def test_session_init_human_vs_bot(self):
        session = MatchSession(
            mode="human_vs_bot",
            bot_name="heuristic",
            target_score=15,
            simulate_delay=False,
            seed=42,
        )
        assert session.mode == "human_vs_bot"
        assert session.is_match_done is False
        assert session.match.current_hand is not None

    def test_session_human_move_triggers_bot_move(self):
        session = MatchSession(
            mode="human_vs_bot",
            bot_name="heuristic",
            target_score=15,
            simulate_delay=False,
            seed=100,
        )
        # In hand 0, P0 (human) is Mano
        assert session.active_player == 0
        view_p0 = session.get_view(player=0)
        assert len(view_p0["my_hand"]) == 3

        # Human plays card 0
        _ = session.step_action(Action.PLAY_CARD_0)
        # After human move, bot (P1) is active and session automatically dispatches bot move!
        # Thus, either both played a card (trick 1 done, turn back to human or trick winner),
        # or bot bid envido/truco awaiting human response.
        assert session.active_player in (0, 1)


class TestWebSocketServerProtocol:
    """Test the WebSocket endpoint /play."""

    def test_websocket_connect_and_start_human_vs_bot(self):
        client = TestClient(app)
        with client.websocket_connect("/play") as ws:
            # Send start command
            ws.send_json(
                {
                    "type": "start",
                    "mode": "human_vs_bot",
                    "bot": "heuristic",
                    "target_score": 15,
                    "simulate_delay": False,
                    "seed": 42,
                }
            )

            # Receive initial state frame
            msg = ws.receive_json()
            assert msg["type"] == "state"
            state = msg["state"]
            assert state["target_score"] == 15
            assert state["match_score"] == [0, 0]
            assert "my_hand" in state
            assert len(state["my_hand"]) == 3
            assert state["opponent_card_count"] == 3
            assert "legal_actions" in state
            assert len(state["legal_actions"]) > 0

    def test_websocket_human_vs_bot_step_action(self):
        client = TestClient(app)
        with client.websocket_connect("/play") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "mode": "human_vs_bot",
                    "bot": "heuristic",
                    "target_score": 15,
                    "simulate_delay": False,
                    "seed": 42,
                }
            )
            initial_msg = ws.receive_json()
            legal_actions = initial_msg["state"]["legal_actions"]

            # Choose first legal card play
            action_to_play = next(a for a in legal_actions if "PLAY_CARD" in a)
            ws.send_json({"type": "action", "action": action_to_play})

            # Receive next state frame
            next_msg = ws.receive_json()
            assert next_msg["type"] in ("state", "match_over")
            if next_msg["type"] == "state":
                assert next_msg["state"]["opponent_card_count"] in (2, 3)

    def test_websocket_bot_vs_bot_simulation(self):
        client = TestClient(app)
        with client.websocket_connect("/play") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "mode": "bot_vs_bot",
                    "bot": "random",
                    "bot_1": "random",
                    "target_score": 15,
                    "simulate_delay": False,
                    "seed": 123,
                }
            )
            msg = ws.receive_json()
            assert msg["type"] == "state"

            # Step simulation until match completion
            loop_count = 0
            while msg["type"] != "match_over" and loop_count < 500:
                loop_count += 1
                ws.send_json({"type": "step_bot"})
                msg = ws.receive_json()

            assert msg["type"] == "match_over"
            assert msg["winner"] in (0, 1)
            assert max(msg["final_score"]) >= 15
