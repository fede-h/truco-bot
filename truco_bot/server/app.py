from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from truco_bot.server.controller import MatchSession, discover_agents

app = FastAPI(title="Truco Bot API")
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def get_index():
    return FileResponse(static_dir / "index.html")


@app.get("/agents")
def get_agents():
    return {"agents": discover_agents()}


@app.websocket("/play")
async def websocket_play(websocket: WebSocket) -> None:
    await websocket.accept()
    session: MatchSession | None = None

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "start":
                mode = data.get("mode", "human_vs_bot")
                bot = data.get("bot", data.get("bot_name", "heuristic"))
                bot_1 = data.get("bot_1")
                target_score = data.get("target_score", 30)
                simulate_delay = data.get("simulate_delay", True)
                seed = data.get("seed")

                session = MatchSession(
                    mode=mode,
                    bot_name=bot,
                    bot_1=bot_1,
                    target_score=target_score,
                    simulate_delay=simulate_delay,
                    seed=seed,
                )

                await websocket.send_json(
                    {
                        "type": "state",
                        "state": session.get_view(player=0),
                        "events": session.pop_events(),
                    }
                )

            elif msg_type == "action":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "Session not started"})
                    continue

                session.step_action(data["action"])
                if session.is_match_done:
                    await websocket.send_json(
                        {
                            "type": "match_over",
                            "winner": session.match_winner,
                            "final_score": [
                                session.match.state.match_score_p0,
                                session.match.state.match_score_p1,
                            ],
                            "events": session.pop_events(),
                        }
                    )
                else:
                    await websocket.send_json(
                        {
                            "type": "state",
                            "state": session.get_view(player=0),
                            "events": session.pop_events(),
                        }
                    )

            elif msg_type == "next_hand":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "Session not started"})
                    continue

                session.next_hand()
                await websocket.send_json(
                    {
                        "type": "state",
                        "state": session.get_view(player=0),
                        "events": session.pop_events(),
                    }
                )

            elif msg_type == "step_bot":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "Session not started"})
                    continue

                session.step_bot()
                if session.is_match_done:
                    await websocket.send_json(
                        {
                            "type": "match_over",
                            "winner": session.match_winner,
                            "final_score": [
                                session.match.state.match_score_p0,
                                session.match.state.match_score_p1,
                            ],
                            "events": session.pop_events(),
                        }
                    )
                else:
                    await websocket.send_json(
                        {
                            "type": "state",
                            "state": session.get_view(player=0),
                            "events": session.pop_events(),
                        }
                    )

            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    }
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:  # noqa: BLE001, S110
            pass
