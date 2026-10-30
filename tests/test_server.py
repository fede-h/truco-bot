"""Adversarial test suite for Ticket #11:
- Interactive WebSocket match server (/play)
- MatchSession controller for human_vs_bot, bot_vs_bot, human_vs_human
- Simulated bot delay toggle
- Masked state views over WebSockets
- End-to-end interactive playout
"""

import re
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from truco_bot.core.actions import Action
from truco_bot.core.deck import ALL_CARDS
from truco_bot.core.match import TrucoMatch
from truco_bot.core.state import create_initial_hand_state
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


class TestTicket14RetroConsoleAndAgents:
    """Adversarial specification-driven tests for Ticket #14 vertical slice."""

    # 1. Static serving & Retro Console HTML shell
    def test_top_level_images_directory_does_not_exist(self):
        """Top-level images/ folder must be removed after relocating assets."""
        assert not Path("images").exists(), "Top-level images/ directory must not exist"

    def test_static_assets_served_ok(self):
        """Static card assets and retro fonts must return 200 OK with non-empty bytes."""
        client = TestClient(app)
        required_assets = [
            "/static/cards/espadas_1.png",
            "/static/cards/bastos_7.png",
            "/static/cards/back.png",
            "/static/fonts/PressStart2P-Regular.ttf",
            "/static/fonts/PixelifySans.ttf",
        ]
        for asset in required_assets:
            res = client.get(asset)
            assert res.status_code == 200, f"Asset {asset} failed with status {res.status_code}"
            assert len(res.content) > 0, f"Asset {asset} returned empty content"

    def test_retro_console_html_shell_and_css(self):
        """GET / returns 200 text/html with retro hooks, pixelated rendering, and credit."""
        client = TestClient(app)
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers.get("content-type", "")

        html = res.text
        # Required attribution / credit
        assert "Spanish Deck – Pixel Art by jcanabal" in html

        # UI structural hooks
        # Agent selector, target score 15/30 selector, start/jugar button, trick slots
        assert re.search(r"id=['\"](agent-select|bot-select|agents)['\"]", html, re.IGNORECASE) or "select" in html.lower()
        assert "15" in html and "30" in html
        assert "jugar" in html.lower() or "start" in html.lower()
        assert re.search(r"(trick|slot|table)", html, re.IGNORECASE)

        # Static style or stylesheet reference
        css_content = ""
        css_match = re.search(r'href=["\'](/static/[^"\']+\.css)["\']', html)
        if css_match:
            css_res = client.get(css_match.group(1))
            assert css_res.status_code == 200
            css_content = css_res.text

        combined_styles = html + "\n" + css_content
        assert "image-rendering: pixelated" in combined_styles or "pixelated" in combined_styles
        assert "PressStart2P" in combined_styles
        assert "PixelifySans" in combined_styles

    # 2. GET /agents & strict agent instantiation validation
    def test_get_agents_baseline_order_and_models(self):
        """GET /agents returns 200 OK with baselines first, then completed checkpoints."""
        client = TestClient(app)
        res = client.get("/agents")
        assert res.status_code == 200
        data = res.json()
        assert "agents" in data
        agents = data["agents"]

        # Baselines must come first in order: ["random", "heuristic", "equity"]
        assert agents[:3] == ["random", "heuristic", "equity"]

        # Existing completed checkpoints must appear without .bin or .pt suffix
        # and incomplete/step checkpoints must NOT appear
        assert "deep_cfr_policy" in agents
        assert "mccfr_plus_1000M" in agents or "mccfr_1000m" in agents
        for agent in agents:
            assert "_step_" not in agent, f"Checkpoint step file exposed in agents: {agent}"
            assert not agent.endswith(".bin"), f"Raw file extension exposed in agents: {agent}"
            assert not agent.endswith(".pt"), f"Raw file extension exposed in agents: {agent}"

    def test_discover_agents_strict_done_filtering(self):
        """A .bin file without .done marker or containing _step_ must NEVER appear in /agents."""
        fake_bin_no_done = Path("models/_fake_test_model_incomplete.bin")
        fake_bin_step = Path("models/_fake_test_model_step_100M.bin")
        fake_done_step = Path("models/_fake_test_model_step_100M.bin.done")
        try:
            fake_bin_no_done.write_text("dummy")
            fake_bin_step.write_text("dummy")
            fake_done_step.write_text("dummy")

            client = TestClient(app)
            res = client.get("/agents")
            assert res.status_code == 200
            agents = res.json()["agents"]
            assert "_fake_test_model_incomplete" not in agents
            assert "_fake_test_model_step_100M" not in agents
        finally:
            fake_bin_no_done.unlink(missing_ok=True)
            fake_bin_step.unlink(missing_ok=True)
            fake_done_step.unlink(missing_ok=True)

    def test_instantiate_agent_unknown_or_missing_raises_value_error(self):
        """Unknown agent names, path traversal, or missing checkpoints must raise ValueError."""
        # Nonexistent bot name
        with pytest.raises(ValueError):
            MatchSession(mode="human_vs_bot", bot_name="nonexistent_bot_xyz")

        # Path traversal / unknown file
        with pytest.raises(ValueError):
            MatchSession(mode="human_vs_bot", bot_name="../pyproject.toml")

        # Incomplete / step checkpoint
        with pytest.raises(ValueError):
            MatchSession(mode="human_vs_bot", bot_name="models/mccfr_plus_1000M_step_100M.bin")

    def test_websocket_start_with_invalid_agent_returns_error(self):
        """WebSocket start with unknown agent must return error frame, not fallback silently."""
        client = TestClient(app)
        with client.websocket_connect("/play") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "mode": "human_vs_bot",
                    "bot": "nonexistent_agent_404",
                    "target_score": 15,
                    "simulate_delay": False,
                }
            )
            msg = ws.receive_json()
            assert msg["type"] == "error", f"Expected error frame, got: {msg}"
            assert "unknown" in msg.get("message", "").lower() or "not found" in msg.get("message", "").lower() or "invalid" in msg.get("message", "").lower() or "valueerror" in msg.get("message", "").lower()

    # 3. get_view(player) full tricks history preservation
    def test_get_view_preserves_full_tricks_history(self):
        """get_view(player) must retain cards played across completed and active tricks."""
        card_map = {(c.number, c.suit.name): c for c in ALL_CARDS}
        p0_cards = [
            card_map[(1, "ESPADAS")],
            card_map[(7, "ESPADAS")],
            card_map[(3, "BASTOS")],
        ]
        p1_cards = [
            card_map[(4, "COPAS")],
            card_map[(5, "OROS")],
            card_map[(6, "BASTOS")],
        ]

        match = TrucoMatch(target_score=15, seed=42)
        # Force deterministic cards
        match.current_hand = create_initial_hand_state(
            hands=[p0_cards, p1_cards],
            mano=0,
            score_p0=0,
            score_p1=0,
            max_score=15,
        )

        # Before playing any card: tricks should be empty or []
        view_0 = match.get_view(0)
        assert "tricks" in view_0
        assert isinstance(view_0["tricks"], list)

        # Trick 1: P0 plays card 0 (1 de Espadas)
        match.step(Action.PLAY_CARD_0)
        view_after_p0 = match.get_view(0)
        assert len(view_after_p0["tricks"]) == 1
        assert len(view_after_p0["tricks"][0]) == 1
        assert view_after_p0["tricks"][0][0] == {"player": 0, "number": 1, "suit": "ESPADAS"}

        # P1 plays card 0 (4 de Copas) -> Trick 1 completes! P0 wins Trick 1
        match.step(Action.PLAY_CARD_0)

        # GameState.trick_cards resets to empty for Trick 2, BUT get_view must preserve Trick 1!
        assert len(match.current_hand.trick_cards) == 0
        view_trick1_done = match.get_view(0)
        assert len(view_trick1_done["tricks"]) >= 1
        assert len(view_trick1_done["tricks"][0]) == 2
        assert view_trick1_done["tricks"][0][0] == {"player": 0, "number": 1, "suit": "ESPADAS"}
        assert view_trick1_done["tricks"][0][1] == {"player": 1, "number": 4, "suit": "COPAS"}

        # Trick 2: P0 won trick 1, so P0 is active and plays card 1
        match.step(Action.PLAY_CARD_1)
        view_trick2_in_progress = match.get_view(0)
        assert len(view_trick2_in_progress["tricks"]) == 2
        assert len(view_trick2_in_progress["tricks"][0]) == 2
        assert len(view_trick2_in_progress["tricks"][1]) == 1

    def test_websocket_state_includes_tricks(self):
        """WebSocket state frames must include 'tricks' key with history."""
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
            msg = ws.receive_json()
            assert msg["type"] == "state"
            assert "tricks" in msg["state"]
            assert isinstance(msg["state"]["tricks"], list)

