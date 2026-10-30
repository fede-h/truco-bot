let ws = null;
let currentState = null;
let isReplaying = false;

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

function showBubble(player, text, durationMs = 1500) {
  const id = player === 0 ? "human-bubble" : "bot-bubble";
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), durationMs);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function processEvents(events) {
  if (!events || events.length === 0) return;
  isReplaying = true;
  for (const ev of events) {
    if (ev.type === "action") {
      const text = SPEECH_CALLS[ev.action];
      if (text) {
        showBubble(ev.player, text, 800);
      }
    } else if (ev.type === "envido_rollout") {
      showBubble(ev.mano_player, "Tengo " + ev.mano_points, 800);
      await sleep(600);
      const pieText =
        ev.winner === ev.pie_player
          ? "¡Y son mejores! (" + ev.pie_points + ")"
          : "Son buenas";
      showBubble(ev.pie_player, pieText, 800);
    }
    await sleep(600);
  }
  isReplaying = false;
}

function connectWebSocket(bot, targetScore) {
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
    if (msg.events && msg.events.length > 0) {
      await processEvents(msg.events);
    }
    if (msg.type === "state") {
      renderState(msg.state);
    } else if (msg.type === "match_over") {
      alert(`Partida terminada! Ganador: ${msg.winner === 0 ? "Vos" : "Bot"}`);
      location.reload();
    } else if (msg.type === "error") {
      alert(`Error: ${msg.message}`);
    }
  };
}

function renderState(state) {
  currentState = state;

  document.getElementById("score-vos").textContent = state.match_score[0];
  document.getElementById("score-bot").textContent = state.match_score[1];
  const half = state.target_score / 2;
  const inBuenas = state.match_score[0] >= half || state.match_score[1] >= half;
  document.getElementById("badge-score").textContent = inBuenas
    ? "BUENAS"
    : "MALAS";
  document.getElementById("badge-mano").textContent =
    state.mano === 0 ? "TU MANO" : "MANO BOT";
  document.getElementById("hud-truco-level").textContent = state.truco_level;

  // Bot hand
  const botHand = document.getElementById("bot-hand");
  botHand.innerHTML = "";
  for (let i = 0; i < state.opponent_card_count; i++) {
    const img = document.createElement("img");
    img.src = "/static/cards/back.png";
    img.className = "card-sprite";
    botHand.appendChild(img);
  }

  // Tricks
  for (let t = 0; t < 3; t++) {
    const slot = document.getElementById(`trick-${t}`);
    const cardsDiv = slot.querySelector(".trick-cards");
    const badge = slot.querySelector(".trick-badge");
    cardsDiv.innerHTML = "";
    badge.textContent = "";

    if (state.tricks && state.tricks[t]) {
      state.tricks[t].forEach((c) => {
        const img = document.createElement("img");
        img.src = cardUrl(c);
        img.className = "card-sprite";
        cardsDiv.appendChild(img);
      });
    }

    if (state.trick_results && state.trick_results[t] !== undefined) {
      const res = state.trick_results[t];
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

document.getElementById("start-btn").onclick = () => {
  const bot = document.getElementById("agent-select").value;
  const targetScore = parseInt(
    document.querySelector('input[name="target_score"]:checked').value,
    10
  );
  document.getElementById("menu-screen").classList.add("hidden");
  document.getElementById("table-screen").classList.remove("hidden");
  connectWebSocket(bot, targetScore);
};

window.onload = initMenu;
