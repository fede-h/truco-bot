let ws = null;
let currentState = null;
let isReplaying = false;

const BOT_ACTION_DELAY_MS = 1000;

const ACTION_LABELS = {
  ENVIDO: "Envido",
  REAL_ENVIDO: "Real Envido",
  FALTA_ENVIDO: "Falta Envido",
  TRUCO: "Truco",
  RETRUCO: "Retruco",
  VALE_CUATRO: "Vale 4",
  QUIERO_ENVIDO: "Quiero",
  QUIERO_TRUCO: "Quiero",
  NO_QUIERO_ENVIDO: "No Quiero",
  NO_QUIERO_TRUCO: "No Quiero",
  IR_AL_MAZO: "Al Mazo",
};

const SPEECH_CALLS = {
  ENVIDO: "¡ENVIDO!",
  REAL_ENVIDO: "¡REAL ENVIDO!",
  FALTA_ENVIDO: "¡FALTA ENVIDO!",
  TRUCO: "¡TRUCO!",
  RETRUCO: "¡RETRUCO!",
  VALE_CUATRO: "¡VALE CUATRO!",
  QUIERO_ENVIDO: "¡QUIERO!",
  QUIERO_TRUCO: "¡QUIERO!",
  NO_QUIERO_ENVIDO: "¡NO QUIERO!",
  NO_QUIERO_TRUCO: "¡NO QUIERO!",
  IR_AL_MAZO: "Me voy al mazo",
};

async function initMenu() {
  const res = await fetch("/agents");
  const data = await res.json();
  const select = document.getElementById("agent-select");
  select.innerHTML = "";
  data.agents.forEach((a) => {
    const opt = document.createElement("option");
    opt.value = a;
    opt.textContent = a;
    select.appendChild(opt);
  });
}

function cardUrl(card) {
  if (!card) return "/static/cards/back.png";
  return `/static/cards/${card.suit.toLowerCase()}_${card.number}.png`;
}

const bubbleTimers = { 0: null, 1: null };

function showBubble(player, text, durationMs = 1000) {
  const id = player === 0 ? "human-bubble" : "bot-bubble";
  const el = document.getElementById(id);
  if (!el) return;
  if (bubbleTimers[player]) {
    clearTimeout(bubbleTimers[player]);
    bubbleTimers[player] = null;
  }
  el.textContent = text;
  el.classList.remove("hidden");
  if (durationMs !== null) {
    bubbleTimers[player] = setTimeout(() => el.classList.add("hidden"), durationMs);
  }
}

function hideBubble(player) {
  const id = player === 0 ? "human-bubble" : "bot-bubble";
  const el = document.getElementById(id);
  if (!el) return;
  if (bubbleTimers[player]) {
    clearTimeout(bubbleTimers[player]);
    bubbleTimers[player] = null;
  }
  el.classList.add("hidden");
}

function syncPendingBotBubble(state) {
  if (!state) return;
  if (state.envido_pending && state.active_player === 0 && state.envido_chain && state.envido_chain.length > 0) {
    const lastCall = state.envido_chain[state.envido_chain.length - 1];
    const text = SPEECH_CALLS[lastCall] || lastCall;
    showBubble(1, text, null);
    return;
  }
  if (state.truco_pending && state.active_player === 0) {
    const trucoCall =
      state.truco_level === 1
        ? "TRUCO"
        : state.truco_level === 2
        ? "RETRUCO"
        : "VALE_CUATRO";
    showBubble(1, SPEECH_CALLS[trucoCall], null);
    return;
  }
  if (!bubbleTimers[1]) {
    hideBubble(1);
  }
}

function cloneState(s) {
  return s ? JSON.parse(JSON.stringify(s)) : null;
}

function buildPreBotState(prevState, finalState, events) {
  const s = cloneState(finalState);
  if (!s) return null;
  s.legal_actions = [];

  const botCardsPlayed = events.filter(
    (e) => e.type === "action" && e.player === 1 && e.card
  ).length;
  if (botCardsPlayed > 0) {
    s.opponent_card_count = (finalState.opponent_card_count || 0) + botCardsPlayed;
    s.tricks = prevState && prevState.tricks ? cloneState(prevState.tricks) : [];
    s.trick_results =
      prevState && prevState.trick_results ? cloneState(prevState.trick_results) : [];
  }

  events.forEach((ev) => {
    if (ev.type === "action" && ev.player === 0 && ev.card) {
      if (!s.tricks.length || s.tricks[s.tricks.length - 1].length === 2) {
        s.tricks.push([]);
      }
      s.tricks[s.tricks.length - 1].push({
        player: 0,
        number: ev.card.number,
        suit: ev.card.suit,
      });
    }
  });
  return s;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function processEvents(events, prevState, finalState) {
  if (!events || events.length === 0) return;
  isReplaying = true;

  let workingState = buildPreBotState(prevState, finalState, events);
  if (workingState) {
    renderBoard(workingState);
  }

  for (const ev of events) {
    if (ev.type === "action") {
      if (ev.player === 0) {
        hideBubble(1);
        const text = SPEECH_CALLS[ev.action];
        if (text) {
          showBubble(0, text, 1000);
        }
      } else {
        // Every bot action takes 1 second (1000 ms)
        await sleep(BOT_ACTION_DELAY_MS);
        const text = SPEECH_CALLS[ev.action];
        if (text) {
          const isPendingShout =
            ev.action === "ENVIDO" ||
            ev.action === "REAL_ENVIDO" ||
            ev.action === "FALTA_ENVIDO" ||
            ev.action === "TRUCO" ||
            ev.action === "RETRUCO" ||
            ev.action === "VALE_CUATRO";
          showBubble(1, text, isPendingShout ? null : 1000);
        } else if (ev.card && workingState) {
          if (
            !workingState.tricks.length ||
            workingState.tricks[workingState.tricks.length - 1].length === 2
          ) {
            workingState.tricks.push([]);
          }
          workingState.tricks[workingState.tricks.length - 1].push({
            player: 1,
            number: ev.card.number,
            suit: ev.card.suit,
          });
          workingState.opponent_card_count = Math.max(
            0,
            workingState.opponent_card_count - 1
          );
          const doneIdx = workingState.tricks.length - 1;
          if (
            workingState.tricks[doneIdx].length === 2 &&
            finalState &&
            finalState.trick_results &&
            finalState.trick_results[doneIdx] !== undefined
          ) {
            workingState.trick_results[doneIdx] = finalState.trick_results[doneIdx];
          }
          renderBoard(workingState);
        }
      }
    } else if (ev.type === "envido_rollout") {
      showBubble(ev.mano_player, "Tengo " + ev.mano_points, 1000);
      await sleep(BOT_ACTION_DELAY_MS);
      const pieText =
        ev.winner === ev.pie_player
          ? "¡Y son mejores! (" + ev.pie_points + ")"
          : "Son buenas";
      showBubble(ev.pie_player, pieText, 1000);
      await sleep(BOT_ACTION_DELAY_MS);
    }
  }
  isReplaying = false;
}

let currentBot = "heuristic";
let currentTargetScore = 30;

// Match-over overlay buttons: "REVANCHA" and "CAMBIAR AGENTE"
function showMatchOver(winner, finalScore) {
  const overlay = document.getElementById("match-over-overlay");
  const title = document.getElementById("match-over-title");
  const score = document.getElementById("match-over-score");
  if (!overlay || !title || !score) return;

  title.textContent = winner === 0 ? "¡GANASTE!" : "PERDISTE";
  score.textContent = `VOS: ${finalScore[0]} - BOT: ${finalScore[1]}`;
  overlay.classList.remove("hidden");
}

function connectWebSocket(bot, targetScore) {
  currentBot = bot;
  currentTargetScore = targetScore;
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${window.location.host}/play`);

  ws.onopen = () => {
    ws.send(
      JSON.stringify({
        type: "start",
        mode: "human_vs_bot",
        bot: bot,
        target_score: targetScore,
        simulate_delay: false,
      })
    );
  };

  ws.onmessage = async (event) => {
    const msg = JSON.parse(event.data);
    const prevState = currentState;
    if (msg.events && msg.events.length > 0) {
      await processEvents(msg.events, prevState, msg.state || prevState);
    }
    if (msg.type === "state") {
      await renderState(msg.state);
    } else if (msg.type === "match_over") {
      showMatchOver(msg.winner, msg.final_score);
    } else if (msg.type === "error") {
      console.error("Server error:", msg.message);
      const errEl = document.getElementById("menu-error");
      if (errEl) {
        errEl.textContent = msg.message;
        errEl.classList.remove("hidden");
      }
      document.getElementById("table-screen").classList.add("hidden");
      document.getElementById("menu-screen").classList.remove("hidden");
    }
  };
}

function renderBoard(state) {
  document.getElementById("score-vos").textContent = state.match_score[0];
  document.getElementById("score-bot").textContent = state.match_score[1];
  const half = state.target_score / 2;
  const inBuenas = state.match_score[0] >= half || state.match_score[1] >= half;
  document.getElementById("badge-score").textContent = inBuenas
    ? "BUENAS"
    : "MALAS";
  document.getElementById("badge-mano").textContent =
    state.mano === 0 ? "TU MANO" : "MANO BOT";

  // Bot hand: revealed opponent cards face-up, rest face-down
  const botHand = document.getElementById("bot-hand");
  botHand.innerHTML = "";
  const revealed = state.revealed_opponent_cards || [];
  revealed.forEach((c) => {
    const img = document.createElement("img");
    img.src = cardUrl(c);
    img.className = "card-sprite";
    botHand.appendChild(img);
  });
  const unrevealedCount = Math.max(
    0,
    state.opponent_card_count - revealed.length
  );
  for (let i = 0; i < unrevealedCount; i++) {
    const img = document.createElement("img");
    img.src = "/static/cards/back.png";
    img.className = "card-sprite";
    botHand.appendChild(img);
  }

  // Tricks: cards stacked one on top of another with slight margin, winner on top
  for (let t = 0; t < 3; t++) {
    const slot = document.getElementById(`trick-${t}`);
    const cardsDiv = slot.querySelector(".trick-cards");
    const badge = slot.querySelector(".trick-badge");
    cardsDiv.innerHTML = "";
    badge.textContent = "";

    const res =
      state.trick_results && state.trick_results[t] !== undefined
        ? state.trick_results[t]
        : undefined;
    const winningPlayer = res === 1 ? 0 : res === -1 ? 1 : null;

    if (state.tricks && state.tricks[t]) {
      state.tricks[t].forEach((c, idx) => {
        const img = document.createElement("img");
        img.src = cardUrl(c);
        img.className = `card-sprite ${idx === 0 ? "card-first" : "card-second"}`;
        if (winningPlayer !== null && c.player === winningPlayer) {
          img.classList.add("card-winner");
        }
        cardsDiv.appendChild(img);
      });
    }

    if (res !== undefined) {
      badge.textContent = res === 1 ? "VOS" : res === -1 ? "BOT" : "PARDA";
    }
  }

  // Human hand
  const humanHand = document.getElementById("human-hand");
  humanHand.innerHTML = "";
  const legals = (!isReplaying && state.legal_actions) || [];

  state.my_hand.forEach((card, idx) => {
    const img = document.createElement("img");
    img.src = cardUrl(card);
    img.className = "card-sprite";

    const actionName = `PLAY_CARD_${idx}`;
    if (legals.includes(actionName)) {
      img.classList.add("card-playable");
      img.onclick = () => {
        if (isReplaying) return;
        ws.send(JSON.stringify({ type: "action", action: actionName }));
      };
    }
    humanHand.appendChild(img);
  });

  // Action bar (only legal call actions)
  const actionBar = document.getElementById("action-bar");
  actionBar.innerHTML = "";
  legals.forEach((action) => {
    const label = ACTION_LABELS[action];
    if (!label) return;
    const btn = document.createElement("button");
    btn.className = "btn-action";
    btn.textContent = label;
    btn.onclick = () => {
      if (isReplaying) return;
      ws.send(JSON.stringify({ type: "action", action: action }));
    };
    actionBar.appendChild(btn);
  });
}

async function renderState(state) {
  currentState = state;
  renderBoard(state);
  syncPendingBotBubble(state);

  if (state.is_match_done) {
    showMatchOver(state.match_winner, state.match_score);
    return;
  }

  if (state.is_hand_done) {
    if (
      state.hand_winner !== null &&
      state.hand_winner !== undefined &&
      state.hand_points_won > 0
    ) {
      showBubble(state.hand_winner, `+${state.hand_points_won}`, 2000);
    }
    await sleep(2000);
    hideBubble(0);
    hideBubble(1);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "next_hand" }));
    }
  }
}

document.getElementById("start-btn").onclick = () => {
  const errEl = document.getElementById("menu-error");
  if (errEl) errEl.classList.add("hidden");
  const bot = document.getElementById("agent-select").value;
  const targetScore = parseInt(
    document.querySelector('input[name="target_score"]:checked').value,
    10
  );
  document.getElementById("match-over-overlay").classList.add("hidden");
  document.getElementById("menu-screen").classList.add("hidden");
  document.getElementById("table-screen").classList.remove("hidden");
  connectWebSocket(bot, targetScore);
};

document.getElementById("revancha-btn").onclick = () => {
  document.getElementById("match-over-overlay").classList.add("hidden");
  if (ws) ws.close();
  connectWebSocket(currentBot, currentTargetScore);
};

document.getElementById("menu-btn").onclick = () => {
  document.getElementById("match-over-overlay").classList.add("hidden");
  document.getElementById("table-screen").classList.add("hidden");
  document.getElementById("menu-screen").classList.remove("hidden");
  if (ws) ws.close();
};

window.onload = initMenu;
