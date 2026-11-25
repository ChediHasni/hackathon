async function sendChatMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addUserMessage(text);

  showSpinner(true);
  try {
    const res = await fetch('/api/chat/', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-CSRFToken': getCsrf()},
      body: JSON.stringify({message: text, session_id: chatSessionId})
    });
    const data = await res.json();
    chatSessionId = data.session_id;
    addBotMessage(data.reply || '', data.zones || []);
    updatePrefs(data.profile || {});
    if (data.zones && data.zones.length) renderZoneMapMarkers(data.zones);
  } catch (e) {
    addBotMessage("Erreur réseau : impossible de joindre le serveur.");
  } finally {
    showSpinner(false);
  }
}

function updatePrefs(profile) {
  const container = document.getElementById('pref-chips');
  container.innerHTML = '';
  if (profile.location) {
    addPrefChip('location', 'Lieu', profile.location);
  }
  if (profile.weights) {
    Object.entries(profile.weights).forEach(([cid, level]) => {
      addPrefChip('criterion_' + cid, cid, level);
    });
  }
  if (profile.min_scores) {
    Object.entries(profile.min_scores).forEach(([cid, val]) => {
      addPrefChip('min_' + cid, cid + ' min', '≥' + val);
    });
  }
  document.getElementById('pref-panel').style.display = Object.keys(profile || {}).length ? 'block' : 'none';
}

function addPrefChip(key, label, value) {
  const container = document.getElementById('pref-chips');
  const div = document.createElement('div');
  div.className = 'pref-chip';
  div.dataset.key = key;
  div.innerHTML = '<span>' + escapeHtml(label) + ':</span> <strong>' + escapeHtml(String(value)) + '</strong> <span class="pref-remove" onclick="removePrefChip(\'' + escapeHtml(key) + '\')">×</span>';
  container.appendChild(div);
}

function removePrefChip(key) {
  const el = document.querySelector('#pref-chips [data-key="' + escapeHtml(key) + '"]');
  if (el) el.remove();
}

function renderZoneMapMarkers(zones) {
  clearZoneMarkers();
  chatZoneLayer = L.featureGroup().addTo(map);
  zones.forEach(z => {
    const zone = z.zone;
    const marker = L.circleMarker([zone.lat, zone.lon], {
      radius: 10, color: '#2563eb', weight: 2, fillColor: '#2563eb', fillOpacity: 0.6
    }).bindTooltip('<strong>' + escapeHtml(zone.name) + '</strong><br/>Score: ' + z.match_score.toFixed(0) + '/100');
    marker.addTo(chatZoneLayer);
  });
  if (zones.length) map.fitBounds(chatZoneLayer.getBounds(), {padding: [40, 40]});
}

function clearZoneMarkers() {
  if (chatZoneLayer) { map.removeLayer(chatZoneLayer); chatZoneLayer = null; }
}

function highlightZoneOnMap(zone) {
  map.setView([zone.lat, zone.lon], 14, {animate: true});
}

document.getElementById('btn-chat-send').onclick = sendChatMessage;
document.getElementById('chat-input').addEventListener('keypress', e => {
  if (e.key === 'Enter') sendChatMessage();
});
document.getElementById('btn-pref-reset').onclick = () => {
  document.getElementById('pref-chips').innerHTML = '';
  document.getElementById('pref-panel').style.display = 'none';
  chatSessionId = null;
};