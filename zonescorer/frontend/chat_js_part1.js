// ─── Chat state ───────────────────────────────────────────────
let chatSessionId = null;
let chatZoneLayer = null;
const chatCriteriaColors = {
  greenness: '#16a34a', climate: '#0ea5e9', building_svf: '#6366f1',
  air_quality: '#10b981', heat: '#f59e0b', accessibility: '#ec4899', transit: '#8b5cf6'
};

function toggleChat() {
  const panel = document.getElementById('chat-panel');
  const pref = document.getElementById('pref-panel');
  const showing = panel.style.display !== 'none';
  panel.style.display = showing ? 'none' : 'block';
  pref.style.display = showing ? 'none' : 'block';
  if (!showing) {
    document.getElementById('chat-input').focus();
    if (!chatSessionId) addBotMessage("Bonjour ! Je suis l'assistant ZoneScore.");
  }
}

function addUserMessage(text) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'chat-msg user';
  div.textContent = text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function addBotMessage(text, zones = [], quickReplies = []) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'chat-msg assistant';
  div.innerHTML = '<div>' + escapeHtml(text) + '</div>';
  if (zones.length) {
    const list = document.createElement('div');
    list.className = 'zone-list';
    zones.forEach(z => {
      const card = document.createElement('div');
      card.className = 'zone-card';
      card.onclick = () => highlightZoneOnMap(z.zone);
      const critBits = Object.entries(z.matched_criteria || {}).map(([cid, val]) => {
        return '<span style="color:' + (chatCriteriaColors[cid] || '#666') + ';font-weight:600">' + cid + '</span> ' + val.toFixed(0);
      }).join('  ·  ');
      card.innerHTML = '<div class="zone-info"><div class="zone-name">' + escapeHtml(z.zone.name) + '</div><div class="zone-meta">' + escapeHtml(z.zone.governorate) + ' · ' + escapeHtml(z.zone.delegation) + '</div><div style="margin-top:4px;font-size:.72rem;color:var(--muted)">' + critBits + '</div></div><div class="match-score">' + z.match_score.toFixed(0) + '<small>/100</small></div>';
      list.appendChild(card);
    });
    div.appendChild(list);
  }
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;

  const qrContainer = document.getElementById('chat-quick-replies');
  qrContainer.innerHTML = '';
  quickReplies.forEach(qr => {
    const btn = document.createElement('button');
    btn.className = 'quick-reply';
    btn.textContent = qr;
    btn.onclick = () => { document.getElementById('chat-input').value = qr; sendChatMessage(); };
    qrContainer.appendChild(btn);
  });
}