// popup.js

const YT_API_KEY = 'AIzaSyDNsaUSxHeDAmiXweVWLXBXnjBhwkCNSco';
const API_URL = 'http://13.60.150.198:8000'; // FastAPI default port

document.getElementById('analyze-btn').addEventListener('click', analyze);

function setStatus(msg, show = true) {
  const bar = document.getElementById('status-bar');
  document.getElementById('status-text').textContent = msg;
  bar.classList.toggle('show', show);
}

function showError(msg) {
  const el = document.getElementById('error-box');
  el.textContent = '⚠ ' + msg;
  el.classList.add('show');
}

function clearError() {
  document.getElementById('error-box').classList.remove('show');
}

async function analyze() {
  clearError();

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url = tab.url || '';
  const match = url.match(/[?&]v=([\w-]{11})/);

  if (!match) {
    document.getElementById('empty-state').style.display = 'none';
    showError('Not a YouTube video page. Open a youtube.com/watch?v=... URL first.');
    return;
  }

  const videoId = match[1];
  document.getElementById('video-id-text').textContent = videoId;
  document.getElementById('empty-state').style.display = 'none';
  document.getElementById('results').classList.remove('show');
  document.getElementById('analyze-btn').disabled = true;

  // Reset image panels
  ['pie', 'trend', 'wc'].forEach(id => {
    document.getElementById(`${id}-chart`).style.display = 'none';
    document.getElementById(`${id}-placeholder`).style.display = 'flex';
    document.getElementById(`${id}-placeholder`).textContent =
      id === 'pie' ? 'Loading chart...' : id === 'trend' ? 'Loading trend...' : 'Loading word cloud...';
  });

  setStatus(`Fetching comments for ${videoId}...`);

  try {
    const comments = await fetchComments(videoId);

    if (!comments.length) {
      showError('No comments found for this video.');
      setStatus('', false);
      document.getElementById('analyze-btn').disabled = false;
      return;
    }

    setStatus(`Fetched ${comments.length} comments — running sentiment analysis...`);
    const predictions = await getSentimentPredictions(comments);

    if (!predictions) {
      setStatus('', false);
      document.getElementById('analyze-btn').disabled = false;
      return;
    }

    setStatus('Rendering results...');
    renderDashboard(comments, predictions);

    // Fire image fetches in parallel (non-blocking)
    const counts = getSentimentCounts(predictions);
    fetchPieChart(counts);
    fetchTrendGraph(predictions.map(p => ({ timestamp: p.timestamp, sentiment: parseInt(p.sentiment) })));
    fetchWordCloud(comments.map(c => c.text));

    setStatus('', false);
    document.getElementById('results').classList.add('show');

  } catch (e) {
    showError('Error: ' + e.message);
    setStatus('', false);
  }

  document.getElementById('analyze-btn').disabled = false;
}

async function fetchComments(videoId) {
  let comments = [], pageToken = '';
  try {
    while (comments.length < 500) {
      const res = await fetch(
        `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${YT_API_KEY}`
      );
      const data = await res.json();
      if (data.error) throw new Error(data.error.message);
      (data.items || []).forEach(item => {
        const s = item.snippet.topLevelComment.snippet;
        comments.push({
          text: s.textOriginal,
          timestamp: s.publishedAt,
          authorId: s.authorChannelId?.value || 'Unknown'
        });
      });
      pageToken = data.nextPageToken;
      if (!pageToken) break;
      setStatus(`Fetching comments... (${comments.length} so far)`);
    }
  } catch (e) {
    showError('YouTube API error: ' + e.message);
  }
  return comments;
}

async function getSentimentPredictions(comments) {
  try {
    const res = await fetch(`${API_URL}/predict_with_timestamps`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ comments })
    });
    if (!res.ok) throw new Error(`Backend responded with ${res.status}`);
    return await res.json();
  } catch (e) {
    showError('FastAPI error: ' + e.message);
    return null;
  }
}

function getSentimentCounts(predictions) {
  const c = { '1': 0, '0': 0, '-1': 0 };
  predictions.forEach(p => { if (c[p.sentiment] !== undefined) c[p.sentiment]++; });
  return c;
}

function renderDashboard(comments, predictions) {
  const counts = getSentimentCounts(predictions);
  const total = comments.length;
  const unique = new Set(comments.map(c => c.authorId)).size;
  const totalWords = comments.reduce((s, c) => s + c.text.split(/\s+/).filter(w => w).length, 0);
  const avgLen = (totalWords / total).toFixed(1);
  const rawScore = predictions.reduce((s, p) => s + parseInt(p.sentiment), 0) / total;
  const normScore = (((rawScore + 1) / 2) * 10).toFixed(1);

  // Metrics
  const mg = document.getElementById('metrics-grid');
  const metrics = [
    { label: 'Total Comments', value: total.toLocaleString(), sub: 'collected' },
    { label: 'Unique Commenters', value: unique.toLocaleString(), sub: `${((unique / total) * 100).toFixed(0)}% unique` },
    { label: 'Avg Length', value: avgLen, sub: 'words / comment' },
    { label: 'Sentiment Score', value: normScore + '/10', sub: rawScore > 0.2 ? 'mostly positive' : rawScore < -0.2 ? 'mostly negative' : 'mixed' }
  ];
  mg.innerHTML = metrics.map((m, i) => `
    <div class="metric-card" style="animation-delay:${i * 0.07}s">
      <div class="metric-label">${m.label}</div>
      <div class="metric-value">${m.value}</div>
      <div class="metric-sub">${m.sub}</div>
    </div>`).join('');

  // Sentiment bars
  const bars = document.getElementById('sentiment-bars');
  const labels = {
    '1':  ['Positive', 'var(--positive)'],
    '0':  ['Neutral',  'var(--neutral)'],
    '-1': ['Negative', 'var(--negative)']
  };
  bars.innerHTML = ['1', '0', '-1'].map(k => {
    const pct = total ? ((counts[k] / total) * 100).toFixed(1) : 0;
    const [name, color] = labels[k];
    return `<div class="sbar-row">
      <div class="sbar-meta">
        <span>${name}</span>
        <span style="color:${color}">${counts[k]} (${pct}%)</span>
      </div>
      <div class="sbar-track">
        <div class="sbar-fill" style="width:${pct}%; background:${color}"></div>
      </div>
    </div>`;
  }).join('');

  // Score ring
  const score = parseFloat(normScore);
  const r = 28, cx = 36, cy = 36;
  const circ = 2 * Math.PI * r;
  const dash = (score / 10) * circ;
  const color = score >= 7 ? '#22d3a0' : score >= 4 ? '#f59e0b' : '#ff3c5f';
  document.getElementById('score-ring-svg').innerHTML = `
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#1a1a24" stroke-width="6"/>
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${color}" stroke-width="6"
      stroke-dasharray="${dash} ${circ}" stroke-dashoffset="${circ / 4}" stroke-linecap="round"/>
    <text x="${cx}" y="${cy + 1}" text-anchor="middle" dominant-baseline="central"
      font-family="Syne,sans-serif" font-size="14" font-weight="800" fill="${color}">${normScore}</text>`;

  // Score summary text
  document.getElementById('score-summary').innerHTML =
    `<b style="color:var(--positive)">${counts['1']}</b> positive<br>
     <b style="color:var(--neutral)">${counts['0']}</b> neutral<br>
     <b style="color:var(--negative)">${counts['-1']}</b> negative`;

  // Comments
  const cl = document.getElementById('comment-list');
  cl.innerHTML = predictions.slice(0, 25).map((item, i) => {
    const s = parseInt(item.sentiment);
    const [cls, label] = s === 1 ? ['badge-pos', '+POS'] : s === -1 ? ['badge-neg', '−NEG'] : ['badge-neu', '○NEU'];
    const text = (item.comment || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return `<div class="comment-item" style="animation-delay:${i * 0.03}s">
      <div class="comment-num">${String(i + 1).padStart(2, '0')}</div>
      <div class="comment-text">${text}</div>
      <div class="sentiment-badge ${cls}">${label}</div>
    </div>`;
  }).join('');
}

async function fetchPieChart(counts) {
  try {
    const res = await fetch(`${API_URL}/generate_chart`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sentiment_counts: counts })
    });
    if (!res.ok) throw new Error();
    const blob = await res.blob();
    const img = document.getElementById('pie-chart');
    img.src = URL.createObjectURL(blob);
    img.style.display = 'block';
    document.getElementById('pie-placeholder').style.display = 'none';
  } catch {
    document.getElementById('pie-placeholder').textContent = 'Chart unavailable';
  }
}

async function fetchWordCloud(texts) {
  try {
    const res = await fetch(`${API_URL}/generate_wordcloud`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ comments: texts })
    });
    if (!res.ok) throw new Error();
    const blob = await res.blob();
    const img = document.getElementById('wc-chart');
    img.src = URL.createObjectURL(blob);
    img.style.display = 'block';
    document.getElementById('wc-placeholder').style.display = 'none';
  } catch {
    document.getElementById('wc-placeholder').textContent = 'Word cloud unavailable';
  }
}

async function fetchTrendGraph(sentimentData) {
  try {
    const res = await fetch(`${API_URL}/generate_trend_graph`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sentiment_data: sentimentData })
    });
    if (!res.ok) throw new Error();
    const blob = await res.blob();
    const img = document.getElementById('trend-chart');
    img.src = URL.createObjectURL(blob);
    img.style.display = 'block';
    document.getElementById('trend-placeholder').style.display = 'none';
  } catch {
    document.getElementById('trend-placeholder').textContent = 'Trend graph unavailable';
  }
}
