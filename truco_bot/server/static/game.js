let ws = null;
let currentState = null;

async function initMenu() {
  const res = await fetch('/agents');
  const data = await res.json();
  const select = document.getElementById('agent-select');
  select.innerHTML = '';
  data.agents.forEach(a => {
    const opt = document.createElement('option');
    opt.value = a;
    opt.textContent = a;
    select.appendChild(opt);
  });
}

function cardUrl(card) {
  if (!card) return '/static/cards/back.png';
  return `/static/cards/${card.suit.toLowerCase()}_${card.number}.png`;
}

function connectWebSocket(bot, targetScore) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${window.location.host}/play`);

  ws.onopen = () => {
    ws.send(JSON.stringify({
      type: 'start',
      mode: 'human_vs_bot',
      bot: bot,
      target_score: targetScore,
      simulate_delay: false
    }));
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'state') {
      renderState(msg.state);
    } else if (msg.type === 'match_over') {
      alert(`Partida terminada! Ganador: ${msg.winner === 0 ? 'Vos' : 'Bot'}`);
      location.reload();
    } else if (msg.type === 'error') {
      alert(`Error: ${msg.message}`);
    }
  };
}

function renderState(state) {
  currentState = state;

  document.getElementById('score-vos').textContent = state.match_score[0];
  document.getElementById('score-bot').textContent = state.match_score[1];
  const half = state.target_score / 2;
  const inBuenas = state.match_score[0] >= half || state.match_score[1] >= half;
  document.getElementById('badge-score').textContent = inBuenas ? 'BUENAS' : 'MALAS';
  document.getElementById('badge-mano').textContent = state.mano === 0 ? 'TU MANO' : 'MANO BOT';
  document.getElementById('hud-truco-level').textContent = state.truco_level;

  // Bot hand
  const botHand = document.getElementById('bot-hand');
  botHand.innerHTML = '';
  for (let i = 0; i < state.opponent_card_count; i++) {
    const img = document.createElement('img');
    img.src = '/static/cards/back.png';
    img.className = 'card-sprite';
    botHand.appendChild(img);
  }

  // Tricks
  for (let t = 0; t < 3; t++) {
    const slot = document.getElementById(`trick-${t}`);
    const cardsDiv = slot.querySelector('.trick-cards');
    const badge = slot.querySelector('.trick-badge');
    cardsDiv.innerHTML = '';
    badge.textContent = '';

    if (state.tricks && state.tricks[t]) {
      state.tricks[t].forEach(c => {
        const img = document.createElement('img');
        img.src = cardUrl(c);
        img.className = 'card-sprite';
        cardsDiv.appendChild(img);
      });
    }

    if (state.trick_results && state.trick_results[t] !== undefined) {
      const res = state.trick_results[t];
      badge.textContent = res === 1 ? 'VOS' : res === -1 ? 'BOT' : 'PARDA';
    }
  }

  // Human hand
  const humanHand = document.getElementById('human-hand');
  humanHand.innerHTML = '';
  const legals = state.legal_actions || [];

  state.my_hand.forEach((card, idx) => {
    const img = document.createElement('img');
    img.src = cardUrl(card);
    img.className = 'card-sprite';

    const actionName = `PLAY_CARD_${idx}`;
    if (legals.includes(actionName)) {
      img.classList.add('card-playable');
      img.onclick = () => {
        ws.send(JSON.stringify({ type: 'action', action: actionName }));
      };
    }
    humanHand.appendChild(img);
  });

  // Action bar (non-card actions, never PLAY_CARD_DOWN_*)
  const actionBar = document.getElementById('action-bar');
  actionBar.innerHTML = '';
  legals.forEach(action => {
    if (action.startsWith('PLAY_CARD')) return;
    const btn = document.createElement('button');
    btn.className = 'btn-action';
    btn.textContent = action;
    btn.onclick = () => {
      ws.send(JSON.stringify({ type: 'action', action: action }));
    };
    actionBar.appendChild(btn);
  });
}

document.getElementById('start-btn').onclick = () => {
  const bot = document.getElementById('agent-select').value;
  const targetScore = parseInt(document.querySelector('input[name="target_score"]:checked').value, 10);
  document.getElementById('menu-screen').classList.add('hidden');
  document.getElementById('table-screen').classList.remove('hidden');
  connectWebSocket(bot, targetScore);
};

window.onload = initMenu;
