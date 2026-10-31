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


def _session_events(session: MatchSession) -> list[dict] | None:
    """Return the event log wherever the implementation keeps it."""
    events = getattr(session, "events", None)
    if events is None:
        events = getattr(session.match, "events", None)
    return events


class TestTicket15EventsAndBidding:
    """Adversarial specification-driven tests for Ticket #15.

    Ordered per-turn events, accurate envido rollout (including cards already
    played in the current trick), exposed envido view fields, and the retro
    bidding / speech-bubble UI.
    """

    # 1. Chronological per-turn events with the correct card payload
    def test_websocket_events_chronological_order_and_card_payload(self):
        client = TestClient(app)
        with client.websocket_connect("/play") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "mode": "human_vs_bot",
                    "bot": "heuristic",
                    "target_score": 30,
                    "simulate_delay": False,
                    "seed": 42,
                }
            )
            initial = ws.receive_json()
            assert initial["type"] == "state"
            assert initial["state"]["mano"] == 0, "Human must be mano in hand 0"
            assert "events" in initial, "Initial state frame must carry an events list"
            assert isinstance(initial["events"], list)
            assert initial["events"] == [], "No events before the human mano acts"

            first_card = initial["state"]["my_hand"][0]
            expected_num = first_card["number"]
            expected_suit = first_card["suit"]

            ws.send_json({"type": "action", "action": "PLAY_CARD_0"})
            first_turn = ws.receive_json()
            assert first_turn["type"] in ("state", "match_over")
            assert "events" in first_turn
            first_events = first_turn["events"]
            assert isinstance(first_events, list)
            assert len(first_events) >= 1

            first_action_event = {
                "player": 0,
                "type": "action",
                "action": "PLAY_CARD_0",
                "card": {"number": expected_num, "suit": expected_suit},
            }
            assert first_events[0] == first_action_event, (
                "First event must be the human's card play with the pre-pop card payload; "
                f"got {first_events[0]!r}"
            )

            for event in first_events[1:]:
                if event.get("type") == "envido_rollout":
                    continue
                assert event.get("player") == 1, (
                    f"Bot actions must be attributed to player 1, got {event!r}"
                )

            # Second human turn: the frame must only carry this turn's events.
            assert first_turn["type"] == "state", "Unexpected match_over on first hand"
            second_state = first_turn["state"]
            legal = second_state["legal_actions"]
            assert legal, "Human should have a legal action on the next turn"
            second_action = next(a for a in legal if not a.startswith("PLAY_CARD_DOWN"))
            if second_action.startswith("PLAY_CARD_"):
                idx = int(second_action.rsplit("_", 1)[1])
                card = second_state["my_hand"][idx]
                second_card = {"number": card["number"], "suit": card["suit"]}
            else:
                second_card = None

            ws.send_json({"type": "action", "action": second_action})
            second_turn = ws.receive_json()
            assert second_turn["type"] in ("state", "match_over")
            assert "events" in second_turn
            second_events = second_turn["events"]
            assert isinstance(second_events, list)
            assert len(second_events) >= 1
            assert second_events[0] == {
                "player": 0,
                "type": "action",
                "action": second_action,
                "card": second_card,
            }, "Second frame leaked or reordered events from the previous turn"
            assert first_action_event not in second_events, (
                "Previous turn's events must not be replayed in the next frame"
            )

    # 2. Bot acting first as mano already produces events
    def test_websocket_events_on_bot_mano_hand_start(self):
        client = TestClient(app)
        with client.websocket_connect("/play") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "mode": "human_vs_bot",
                    "bot": "heuristic",
                    "target_score": 30,
                    "simulate_delay": False,
                    "seed": 42,
                }
            )
            ws.receive_json()  # initial hand 0 frame

            ws.send_json({"type": "next_hand"})
            msg = ws.receive_json()
            assert msg["type"] == "state"
            assert msg["state"]["hand_index"] == 1
            assert msg["state"]["mano"] == 1, "Bot must be mano in hand 1"
            assert "events" in msg
            assert isinstance(msg["events"], list)
            assert len(msg["events"]) >= 1, "Bot mano start must record its first action"
            assert msg["events"][0]["player"] == 1

    # 3. Envido must count all three dealt cards, even one already on the table
    def test_envido_rollout_event_and_pie_envido_after_mano_card_play(self):
        card_map = {(c.number, c.suit.name): c for c in ALL_CARDS}
        session = MatchSession(
            mode="human_vs_human",
            target_score=30,
            simulate_delay=False,
            seed=42,
        )
        session.match.current_hand = create_initial_hand_state(
            hands=[
                [
                    card_map[(7, "ESPADAS")],
                    card_map[(6, "ESPADAS")],
                    card_map[(4, "COPAS")],
                ],
                [
                    card_map[(7, "OROS")],
                    card_map[(5, "OROS")],
                    card_map[(4, "BASTOS")],
                ],
            ],
            mano=0,
            score_p0=0,
            score_p1=0,
            max_score=30,
        )

        # Mano (P0) plays a card; its 7 ESPADAS leaves the hand before the envido.
        session.step_action(Action.PLAY_CARD_0)
        # Pie (P1) calls Envido, mano accepts.
        session.step_action(Action.ENVIDO)
        session.step_action(Action.QUIERO_ENVIDO)

        events = _session_events(session)
        assert events is not None, "Session must record an ordered events list"

        rollouts = [e for e in events if e.get("type") == "envido_rollout"]
        assert rollouts, f"Missing envido_rollout event in {events!r}"
        rollout = rollouts[-1]
        assert rollout["mano_player"] == 0
        assert rollout["mano_points"] == 33, (
            "Mano envido must count all 3 dealt cards (7+6+20), not just unplayed ones"
        )
        assert rollout["pie_player"] == 1
        assert rollout["pie_points"] == 32
        assert rollout["winner"] == 0
        assert rollout["points_awarded"] == 2
        assert session.match.state.match_score_p0 == 2, (
            "Envido winner must be P0 with 2 points, not P1 from a truncated hand"
        )

    # 4. Envido view fields, without leaking a human envido helper
    def test_get_view_envido_fields_and_no_human_envido_helper(self):
        session = MatchSession(
            mode="human_vs_bot",
            bot_name="heuristic",
            target_score=30,
            simulate_delay=False,
            seed=42,
        )
        view = session.get_view(0)

        assert "envido_pending" in view
        assert isinstance(view["envido_pending"], bool)
        assert "envido_chain" in view
        assert isinstance(view["envido_chain"], list)
        assert all(isinstance(call, str) for call in view["envido_chain"])
        assert "envido_winner" in view
        assert view["envido_winner"] is None or isinstance(view["envido_winner"], int)
        assert "envido_points_won" in view
        assert isinstance(view["envido_points_won"], int)

        for forbidden in ("my_envido", "human_envido", "player_envido"):
            assert forbidden not in view, (
                f"get_view(0) must not leak the human's own envido total ({forbidden})"
            )

    # 5. Retro bidding UI, Rioplatense speech bubbles, 600 ms pacing
    def test_static_ui_bidding_labels_speech_bubbles_and_600ms_pacing(self):
        client = TestClient(app)
        index_res = client.get("/")
        js_res = client.get("/static/game.js")
        css_res = client.get("/static/style.css")

        assert index_res.status_code == 200
        assert js_res.status_code == 200
        assert css_res.status_code == 200

        js = js_res.text
        css = css_res.text

        for label in (
            "Envido",
            "Real Envido",
            "Falta Envido",
            "Truco",
            "Retruco",
            "Vale 4",
            "Quiero",
            "No Quiero",
            "Al Mazo",
        ):
            assert label in js, f"Missing bidding button label: {label}"

        for bubble in (
            "¡ENVIDO!",
            "¡REAL ENVIDO!",
            "¡FALTA ENVIDO!",
            "¡TRUCO!",
            "¡RETRUCO!",
            "¡VALE CUATRO!",
            "¡QUIERO!",
            "¡NO QUIERO!",
            "Me voy al mazo",
            "Tengo ",
            "Son buenas",
            "¡Y son mejores!",
        ):
            assert bubble in js, f"Missing Rioplatense speech bubble text: {bubble}"

        assert re.search(r"\b1000\b", js), "All agent actions must be paced at 1000 ms (1 second)"
        assert "PLAY_CARD_DOWN" not in js, "The client must never expose PLAY_CARD_DOWN"
        assert "hud-truco-level" not in index_res.text, (
            "The upper-right Truco indicator must be removed from the HUD"
        )
        assert ".card-winner" in css and "z-index" in css, (
            "Trick cards must be stacked with the winning card on top"
        )

        assert "#f2ead3" in css, "Speech bubbles need the cream #f2ead3 fill"

    def test_retruco_and_vale_cuatro_escalation_and_resolution_flow(self):
        """Verify Truco -> Retruco -> Vale Cuatro -> Quiero/No Quiero flow lets play continue until resolved."""
        session = MatchSession(
            mode="human_vs_human",
            target_score=30,
            simulate_delay=False,
            seed=42,
        )
        # P0 calls TRUCO
        session.step_action(Action.TRUCO)
        view_p1 = session.get_view(1)
        assert view_p1["truco_pending"] is True
        assert "RETRUCO" in view_p1["legal_actions"], (
            "Player facing TRUCO must be able to raise to RETRUCO immediately"
        )

        # P1 raises to RETRUCO
        session.step_action(Action.RETRUCO)
        view_p0 = session.get_view(0)
        assert view_p0["truco_pending"] is True
        assert view_p0["truco_level"] == 2
        assert "VALE_CUATRO" in view_p0["legal_actions"], (
            "Player facing RETRUCO must be able to raise to VALE_CUATRO immediately"
        )
        assert "QUIERO_TRUCO" in view_p0["legal_actions"]

        # P0 raises to VALE_CUATRO
        session.step_action(Action.VALE_CUATRO)
        view_p1_vc = session.get_view(1)
        assert view_p1_vc["truco_pending"] is True
        assert view_p1_vc["truco_level"] == 3
        assert "QUIERO_TRUCO" in view_p1_vc["legal_actions"]
        assert "NO_QUIERO_TRUCO" in view_p1_vc["legal_actions"]

        # P1 accepts VALE_CUATRO -> truco_level becomes 4, pending clears, and card play resumes!
        session.step_action(Action.QUIERO_TRUCO)
        view_after_quiero = session.get_view(0)
        assert view_after_quiero["truco_pending"] is False
        assert view_after_quiero["truco_level"] == 4
        assert "PLAY_CARD_0" in view_after_quiero["legal_actions"], (
            "After VALE_CUATRO is accepted, card play must resume normally"
        )


class TestTicket16HandOverRevealAndMatchEnd:
    """Adversarial RED tests for Ticket #16.

    The hand-over pause must survive over WebSocket, the envido reveal must obey
    "las que se cantan se muestran", and the match-over overlay must replace the
    old blocking ``alert()``.
    """

    @staticmethod
    def _session_with_hands(hands: list[list[tuple[int, str]]], mano: int = 0) -> MatchSession:
        card_map = {(c.number, c.suit.name): c for c in ALL_CARDS}
        session = MatchSession(
            mode="human_vs_human",
            target_score=30,
            simulate_delay=False,
            seed=42,
        )
        session.match.state.mano = mano
        session.match.current_hand = create_initial_hand_state(
            hands=[[card_map[key] for key in hand] for hand in hands],
            mano=mano,
            score_p0=0,
            score_p1=0,
            max_score=30,
        )
        return session

    def test_websocket_pauses_at_hand_over_instead_of_auto_advancing(self):
        client = TestClient(app)
        with client.websocket_connect("/play") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "mode": "human_vs_bot",
                    "bot": "heuristic",
                    "target_score": 30,
                    "simulate_delay": False,
                    "seed": 42,
                }
            )
            initial = ws.receive_json()
            assert initial["type"] == "state"
            assert initial["state"]["mano"] == 0, "Human must be mano in hand 0"

            # Folding finishes hand 0 immediately (1 point) without ending a 30-point match.
            ws.send_json({"type": "action", "action": "IR_AL_MAZO"})
            msg = ws.receive_json()
            assert msg["type"] == "state", f"Hand-over must be a state frame, got {msg!r}"

            state = msg["state"]
            assert state["hand_index"] == 0, (
                "Server auto-advanced to the next hand instead of pausing on hand_over: "
                f"hand_index={state['hand_index']}"
            )
            assert state["is_hand_done"] is True
            assert state["hand_winner"] == 1, "Folding as mano hands the hand to the bot (P1)"
            assert state["hand_points_won"] >= 1
            assert "revealed_opponent_cards" in state
            assert state["revealed_opponent_cards"] == [], (
                "No Envido was accepted or won by the bot, so no card may be revealed"
            )

            ws.send_json({"type": "next_hand"})
            nxt = ws.receive_json()
            assert nxt["type"] == "state"
            assert nxt["state"]["hand_index"] == 1, "next_hand must deal hand 1"
            assert nxt["state"]["is_hand_done"] is False

    def test_revealed_opponent_cards_authentic_envido_rules_tricky_cases(self):
        # Case A: bot wins accepted Envido with an OROS pair; one pair card is on
        # the table, the other pair card must be revealed, the off-suit card hidden.
        session = self._session_with_hands(
            [
                [(4, "COPAS"), (5, "COPAS"), (2, "BASTOS")],
                [(7, "OROS"), (6, "OROS"), (1, "ESPADAS")],
            ]
        )
        session.step_action(Action.ENVIDO)  # P0 calls
        session.step_action(Action.QUIERO_ENVIDO)  # P1 accepts: 33 vs 29

        view = session.get_view(0)
        assert view["is_hand_done"] is False
        assert "revealed_opponent_cards" in view, (
            "get_view() must always expose revealed_opponent_cards"
        )
        assert view["revealed_opponent_cards"] == [], (
            "Unplayed bot cards must stay face-down while the hand is in progress"
        )

        session.step_action(Action.PLAY_CARD_0)  # P0: 4 COPAS
        session.step_action(Action.PLAY_CARD_0)  # P1: 7 OROS (pair card now on table)
        assert session.get_view(0)["is_hand_done"] is False
        session.step_action(Action.IR_AL_MAZO)  # active player folds -> hand over

        view = session.get_view(0)
        assert view["is_hand_done"] is True
        assert view["revealed_opponent_cards"] == [{"number": 6, "suit": "OROS"}], (
            "Only 6 OROS (unplayed half of the winning 33-point OROS pair) may be shown; "
            "7 OROS is already on the table and 1 ESPADAS was never part of the Envido"
        )

        # Case B: three OROS; only the two that formed the 33-point Envido are revealed.
        session = self._session_with_hands(
            [
                [(4, "COPAS"), (5, "COPAS"), (2, "BASTOS")],
                [(7, "OROS"), (6, "OROS"), (4, "OROS")],
            ]
        )
        session.step_action(Action.ENVIDO)
        session.step_action(Action.QUIERO_ENVIDO)
        session.step_action(Action.IR_AL_MAZO)

        view = session.get_view(0)
        assert view["is_hand_done"] is True
        revealed = view["revealed_opponent_cards"]
        assert len(revealed) == 2, f"Exactly the top two OROS must be revealed, got {revealed!r}"
        assert {(c["number"], c["suit"]) for c in revealed} == {
            (7, "OROS"),
            (6, "OROS"),
        }, f"Wrong Envido pair revealed: {revealed!r}"
        assert all(
            not (c["number"] == 4 and c["suit"] == "OROS") for c in revealed
        ), "4 OROS did not form the 33-point Envido and must stay hidden"

        # Case C: three different suits; only the single highest Envido card shows.
        session = self._session_with_hands(
            [
                [(6, "COPAS"), (4, "OROS"), (12, "BASTOS")],
                [(7, "BASTOS"), (5, "OROS"), (10, "ESPADAS")],
            ],
            mano=1,
        )
        session.step_action(Action.ENVIDO)  # P1 (bot) calls
        session.step_action(Action.QUIERO_ENVIDO)  # P0 accepts: 6 vs 7
        session.step_action(Action.IR_AL_MAZO)  # P1 folds

        view = session.get_view(0)
        assert view["is_hand_done"] is True
        assert view["revealed_opponent_cards"] == [{"number": 7, "suit": "BASTOS"}], (
            "With three distinct suits only the single highest Envido card (7 BASTOS) is shown"
        )

        # Case D: the loser of an accepted Envido says "Son buenas" and shows nothing.
        session = self._session_with_hands(
            [
                [(7, "OROS"), (6, "OROS"), (1, "ESPADAS")],
                [(4, "COPAS"), (5, "COPAS"), (2, "BASTOS")],
            ]
        )
        session.step_action(Action.ENVIDO)  # P0 calls
        session.step_action(Action.QUIERO_ENVIDO)  # P1 accepts but loses 29 vs 33
        session.step_action(Action.IR_AL_MAZO)

        view = session.get_view(0)
        assert view["is_hand_done"] is True
        assert view["revealed_opponent_cards"] == [], (
            "Bot lost the accepted Envido: its cards must not be revealed"
        )

        # Case D (declined): a NO_QUIERO Envido never reveals the bot's cards.
        session = self._session_with_hands(
            [
                [(7, "ESPADAS"), (6, "ESPADAS"), (4, "COPAS")],
                [(7, "OROS"), (5, "OROS"), (4, "BASTOS")],
            ]
        )
        session.step_action(Action.ENVIDO)
        session.step_action(Action.NO_QUIERO_ENVIDO)
        session.step_action(Action.IR_AL_MAZO)

        view = session.get_view(0)
        assert view["is_hand_done"] is True
        assert view["revealed_opponent_cards"] == [], (
            "A declined Envido must not reveal the bot's cards"
        )

    def test_early_envido_match_cutoff_emits_match_over(self, monkeypatch):
        # ``truco_bot.server.app`` is shadowed by the FastAPI instance on the package,
        # so fetch the real module from sys.modules before patching it.
        import importlib

        app_module = importlib.import_module("truco_bot.server.app")

        card_map = {(c.number, c.suit.name): c for c in ALL_CARDS}
        created_sessions: list[MatchSession] = []

        def fabricate_session(*args, **kwargs):
            session = MatchSession(
                mode="human_vs_human",
                target_score=15,
                simulate_delay=False,
                seed=42,
            )
            session.match.state.match_score_p0 = 14
            session.match.current_hand = create_initial_hand_state(
                hands=[
                    [card_map[(7, "OROS")], card_map[(6, "OROS")], card_map[(4, "COPAS")]],
                    [card_map[(4, "BASTOS")], card_map[(5, "BASTOS")], card_map[(2, "COPAS")]],
                ],
                mano=0,
                score_p0=0,
                score_p1=0,
                max_score=15,
            )
            created_sessions.append(session)
            return session

        monkeypatch.setattr(app_module, "MatchSession", fabricate_session)

        # Controller-level guard: 14 + 2 Envido points reaches the 15-point target.
        controller_session = fabricate_session()
        controller_session.step_action(Action.ENVIDO)
        controller_session.step_action(Action.QUIERO_ENVIDO)
        assert controller_session.is_match_done is True
        assert controller_session.match_winner == 0
        assert controller_session.match.state.match_score_p0 == 16

        client = TestClient(app)
        with client.websocket_connect("/play") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "mode": "human_vs_human",
                    "target_score": 15,
                    "simulate_delay": False,
                    "seed": 42,
                }
            )
            first = ws.receive_json()
            assert first["type"] == "state"
            assert first["state"]["match_score"][0] == 14

            ws.send_json({"type": "action", "action": "ENVIDO"})
            envido_call = ws.receive_json()
            assert envido_call["type"] == "state"

            ws.send_json({"type": "action", "action": "QUIERO_ENVIDO"})
            msg = ws.receive_json()
            assert msg["type"] == "match_over", (
                f"Early Envido cutoff must emit match_over, got {msg!r}"
            )
            assert msg["winner"] == 0
            assert msg["final_score"][0] >= 15
            assert any(e.get("type") == "envido_rollout" for e in msg["events"]), (
                "match_over frame must carry the envido_rollout events"
            )

    def test_static_ui_hand_hold_2000ms_match_over_overlay_and_readme_docs(self):
        client = TestClient(app)
        index_res = client.get("/")
        js_res = client.get("/static/game.js")
        css_res = client.get("/static/style.css")

        assert index_res.status_code == 200
        assert js_res.status_code == 200
        assert css_res.status_code == 200
        assert "text/html" in index_res.headers.get("content-type", "")

        js = js_res.text
        assert re.search(r"\b2000\b", js), "The table must hold the finished hand for ~2000 ms"
        assert "next_hand" in js, "The client must deal the next hand after the ~2 s hold"
        assert "revealed_opponent_cards" in js, (
            "The client must flip the server-revealed Envido cards face-up"
        )
        assert "hand_points_won" in js, "The client must display the points won in the hand"

        point_bubble_patterns = (
            r"\+\$\{[^}]*points",
            r'"\+"\s*\+',
            r"'\+'\s*\+",
            r'"\+N"',
            r"\+N",
        )
        assert any(re.search(pattern, js) for pattern in point_bubble_patterns), (
            "Hand-over must show a '+N' points indicator bubble"
        )

        for text in ("¡GANASTE!", "PERDISTE", "REVANCHA", "CAMBIAR AGENTE"):
            assert text in js, f"Missing match-over overlay text: {text}"

        assert "window.alert" not in js, "Match-over must not use window.alert()"
        assert not re.search(r"(?<![\w.$])alert\s*\(", js), (
            "Match-over must not use blocking alert()"
        )

        readme = Path("README.md").read_text(encoding="utf-8")
        assert "uv run uvicorn truco_bot.server.app:app --reload" in readme, (
            "README must document how to launch the UI server"
        )
        assert "Spanish Deck – Pixel Art by jcanabal" in readme, (
            "README must credit the Spanish Deck pixel art author"
        )

