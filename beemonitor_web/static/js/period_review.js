// Period review (memory/62 §4, §4b): every event of one device or site over
// hours or days on one time axis, one player that moves from clip to clip,
// event review across clips, and review of suggested exit → entry trips.
//
// The page carries the period as JSON (#period-data): clips (each with its
// offset on the axis, frame rate, overlay / stream / review URLs and events)
// and the suggested trips. Votes go to each clip's own review endpoint, so
// the clip page, the batch and the CSVs show them too.
(function () {
  'use strict';
  var root = document.getElementById('period-review');
  var dataEl = document.getElementById('period-data');
  if (!root || !dataEl) return;
  var P = JSON.parse(dataEl.textContent);
  var $ = function (sel, el) { return (el || root).querySelector(sel); };
  var $$ = function (sel, el) { return Array.prototype.slice.call((el || root).querySelectorAll(sel)); };
  var COLORS = {exit: '#0369a1', enter: '#b45309', stay: '#6d28d9'};
  var MAX_LANES = 12;

  var video = $('[data-pr-video]'), canvas = $('[data-pr-canvas]');
  var status = $('[data-pr-status]');
  var ov = window.TrackOverlay ? TrackOverlay(video, canvas) : null;
  var cur = -1;                 // index of the loaded clip
  var pendingFrame = null;      // frame to seek to once the clip has loaded

  // Every event, with the clip it belongs to.
  var all = [];
  P.clips.forEach(function (c, ci) { c.events.forEach(function (e) { all.push({c: ci, e: e}); }); });
  all.sort(function (a, b) { return a.e.t - b.e.t; });

  function pct(t) { return Math.max(0, Math.min(100, 100 * t / (P.span || 1))); }
  function tube(ref) { return String(ref || '').replace(/^reference_/, ''); }
  function clock(iso) { return (iso || '').replace('T', ' ').slice(0, 19); }
  function csrf() { var el = $('[name=csrfmiddlewaretoken]'); return el ? el.value : ''; }
  function post(url, body) {
    return fetch(url, {method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf()},
      body: JSON.stringify(body)}).then(function (r) { return r.json(); })
      .then(function (d) { if (!d.ok) throw new Error(d.error || 'Could not save'); return d; });
  }

  // ── The axis ─────────────────────────────────────────────────────────────
  function renderTicks() {
    var box = $('[data-pr-ticks]');
    box.innerHTML = '';
    var start = new Date(P.start), hours = P.span / 3600;
    var step = hours <= 12 ? 1 : hours <= 48 ? 6 : 24;
    var first = new Date(start); first.setMinutes(0, 0, 0);
    if (first < start) first.setHours(first.getHours() + 1);
    for (var t = first; (t - start) / 1000 <= P.span; t = new Date(t.getTime() + step * 3600e3)) {
      if (step > 1 && t.getHours() % step) continue;
      var s = document.createElement('span');
      s.className = 'absolute -translate-x-1/2 whitespace-nowrap';
      s.style.left = pct((t - start) / 1000) + '%';
      s.textContent = step >= 24 ? (t.getMonth() + 1) + '/' + t.getDate()
        : String(t.getHours()).padStart(2, '0') + ':00';
      box.appendChild(s);
    }
  }

  function renderStrip() {
    var strip = $('[data-pr-strip]');
    strip.innerHTML = '';
    P.clips.forEach(function (c, ci) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'absolute top-0.5 bottom-0.5 rounded-sm';
      b.style.left = pct(c.offset) + '%';
      b.style.width = Math.max(0.3, pct(c.duration)) + '%';
      b.style.background = ci === cur ? '#15803d' : '#6b7280';
      b.title = c.title + ' · ' + clock(c.start);
      b.setAttribute('aria-label', 'Load clip ' + b.title);
      b.addEventListener('click', function () { loadClip(ci, 0, true); });
      strip.appendChild(b);
    });
    var now = document.createElement('div');
    now.setAttribute('data-pr-now', '');
    now.className = 'absolute -top-0.5 -bottom-0.5 w-0.5 bg-gray-900 pointer-events-none';
    strip.appendChild(now);
  }

  var showAll = false;
  function renderLanes() {
    var box = $('[data-pr-lanes]');
    box.innerHTML = '';
    var by = {}, addedBy = {};
    all.forEach(function (x) { (by[x.e.reference] = by[x.e.reference] || []).push(x); });
    P.clips.forEach(function (c, ci) {
      (c.added || []).forEach(function (a) {
        by[a.reference] = by[a.reference] || [];
        (addedBy[a.reference] = addedBy[a.reference] || []).push({c: ci, a: a});
      });
    });
    var order = Object.keys(by).sort(function (a, b) { return by[b].length - by[a].length; });
    var links = $('[data-pr-links]').checked;
    (showAll ? order : order.slice(0, MAX_LANES)).forEach(function (ref) {
      var row = document.createElement('div');
      row.className = 'flex items-center gap-2';
      var name = document.createElement('button');
      name.type = 'button';
      name.className = 'w-[68px] shrink-0 text-right font-mono text-xs font-semibold text-gray-700 hover:underline';
      name.textContent = 'tube ' + tube(ref);
      name.addEventListener('click', function () { renderSide(ref); });
      var track = document.createElement('div');
      track.className = 'relative flex-1 h-4 rounded-sm bg-gray-100';
      if (links) {
        P.trips.forEach(function (tr) {
          if (tr.reference !== ref) return;
          var a = findEvent(tr.exit), b = findEvent(tr.entry);
          if (!a || !b) return;
          var band = document.createElement('span');
          band.className = 'absolute top-1.5 h-1.5 rounded-full';
          band.style.left = pct(a.t) + '%';
          band.style.width = Math.max(0.2, pct(b.t) - pct(a.t)) + '%';
          band.style.background = tr.mine && tr.mine.verdict === 'accepted' ? 'rgba(21,128,61,0.35)'
            : 'rgba(3,105,161,0.18)';
          band.title = 'exit → next entry: ' + Math.round(tr.gap_sec / 60) + ' min' +
            (tr.same_marker ? ' (same marker)' : '');
          track.appendChild(band);
        });
      }
      by[ref].forEach(function (x) {
        var e = x.e, m = document.createElement('button');
        m.type = 'button';
        m.className = 'absolute top-0.5 bottom-0.5 rounded-sm';
        m.style.left = pct(e.t) + '%';
        m.style.width = '3px';
        m.style.background = COLORS[e.action] || '#374151';
        var acc = e.accepted || 0, rej = e.rejected || 0;
        if (acc || rej) m.style.boxShadow = '0 0 0 2px #fff, 0 0 0 4px ' +
          (acc && rej ? '#b45309' : rej ? '#b91c1c' : '#15803d');
        m.title = clock(e.clock) + ' ' + e.action + ' tube ' + tube(ref) + ' · track ' + e.track_id;
        m.setAttribute('aria-label', m.title);
        m.addEventListener('click', function () { goToEvent(x); });
        track.appendChild(m);
      });
      // Added as missed by a reviewer: hollow and dashed (memory/62 §3b).
      (addedBy[ref] || []).forEach(function (y) {
        var a = y.a, m = document.createElement('button');
        m.type = 'button';
        m.className = 'absolute top-0.5 bottom-0.5 rounded-sm bg-white';
        m.style.left = pct(a.t) + '%';
        m.style.width = '6px';
        m.style.border = '2px dashed ' + (COLORS[a.action] || '#374151');
        m.title = a.action + ' tube ' + tube(ref) + ', added by ' + a.by;
        m.setAttribute('aria-label', m.title);
        m.addEventListener('click', function () { loadClip(y.c, a.frame, false); });
        track.appendChild(m);
      });
      var count = document.createElement('span');
      count.className = 'w-[52px] shrink-0 font-mono text-[11px] text-gray-600';
      count.textContent = by[ref].filter(function (x) { return x.e.action === 'exit'; }).length + ' / ' +
        by[ref].filter(function (x) { return x.e.action === 'enter'; }).length;
      row.appendChild(name); row.appendChild(track); row.appendChild(count);
      box.appendChild(row);
    });
    if (order.length > MAX_LANES) {
      var more = document.createElement('button');
      more.type = 'button';
      more.className = 'self-start ml-[76px] text-xs text-amber-700 hover:underline min-h-[28px]';
      more.textContent = showAll ? 'Show the ' + MAX_LANES + ' busiest tubes' : '+ ' + (order.length - MAX_LANES) + ' more tubes';
      more.addEventListener('click', function () { showAll = !showAll; renderLanes(); });
      box.appendChild(more);
    }
    paintProgress();
  }
  $('[data-pr-links]').addEventListener('change', renderLanes);

  function findEvent(ref) {
    var c = P.clips[ref.clip];
    return c && c.events.filter(function (e) { return e.key === ref.key; })[0];
  }

  function renderSide(ref) {
    $('[data-pr-side-title]').textContent = 'Tube ' + tube(ref) + ' across the period';
    var box = $('[data-pr-side]');
    box.innerHTML = '';
    all.filter(function (x) { return x.e.reference === ref; }).forEach(function (x) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'text-left hover:underline';
      b.innerHTML = '<span class="font-mono text-gray-600"></span> <span></span>';
      b.children[0].textContent = clock(x.e.clock).slice(11);
      b.children[1].textContent = x.e.action + ' · track ' + x.e.track_id + ' · ' + P.clips[x.c].title;
      b.addEventListener('click', function () { goToEvent(x); });
      box.appendChild(b);
    });
  }

  function paintProgress() {
    var el = $('[data-pr-progress]');
    var reviewed = all.filter(function (x) { return (x.e.accepted || 0) + (x.e.rejected || 0) > 0; }).length;
    var tripsDone = P.trips.filter(function (t) { return t.mine || t.accepted || t.rejected; }).length;
    el.textContent = all.length + ' events · ' + reviewed + ' reviewed · ' +
      P.trips.length + ' suggested trips · ' + tripsDone + ' reviewed';
  }

  // ── The player ───────────────────────────────────────────────────────────
  function loadClip(ci, frame, play) {
    var c = P.clips[ci];
    if (!c || !ov) return Promise.resolve();
    if (ci === cur) {
      if (frame != null) ov.seekFrame(frame);
      if (play) video.play();
      return Promise.resolve();
    }
    cur = ci;
    pendingFrame = frame;
    $('[data-pr-clip]').textContent = c.title + ' · ' + clock(c.start);
    video.src = c.stream_url;
    renderStrip();
    status.textContent = 'Loading ' + c.title + '…';
    return ov.load(c.overlay_url).then(function () {
      status.textContent = '';
      if (play) { var p = video.play(); if (p && p.catch) p.catch(function () {}); }
    }).catch(function () { status.textContent = 'Could not load the tracks of ' + c.title + '.'; });
  }
  video.addEventListener('loadedmetadata', function () {
    if (pendingFrame != null && ov) { ov.seekFrame(pendingFrame); pendingFrame = null; }
  });
  // At the end of a clip, the next one.
  video.addEventListener('ended', function () {
    if (tp.on) { tp.k++; tpStart(); return; }
    if (rv.on || trip.on || cur + 1 >= P.clips.length) return;
    loadClip(cur + 1, 0, true);
  });
  if (ov) ov.onFrame(function (info) {
    var c = P.clips[cur];
    var now = $('[data-pr-now]');
    if (now && c) now.style.left = pct(c.offset + info.frame / c.fps) + '%';
    if (rv.on && !rv.waiting && rv.target && rv.target.c === cur && info.frame >= rv.target.e.frame) rvArrive();
    if (trip.on) tripHead();
    if (tp.on) {
      var sg = tp.segs[tp.k];
      if (sg && sg.c === cur && info.frame >= sg.to) { tp.k++; tpStart(); }
    }
  });

  // ── Event review across clips ────────────────────────────────────────────
  var rv = {on: false, list: [], i: 0, target: null, waiting: false, ourSeek: false};
  var card = $('[data-pr-card]');
  // An event was clicked (a mark, the side list). In review mode the review
  // goes to it, card and all; otherwise it is a plain jump.
  function goToEvent(x) {
    if (!rv.on) { loadClip(x.c, x.e.frame, false); return; }
    var idx = rv.list.indexOf(x);
    if (idx < 0) { loadClip(x.c, x.e.frame, false); return; }
    rv.i = idx; rv.target = x; rv.waiting = false; rv.ourSeek = true;
    card.hidden = true; $('[data-pr-reasons]').hidden = true;
    loadClip(x.c, x.e.frame, false).then(rvArrive);
  }
  // The viewer scrubbed the video themselves: review carries on to the next
  // event after that point instead of showing a card for one it left.
  video.addEventListener('seeked', function () { rv.ourSeek = false; });
  video.addEventListener('seeking', function () {
    if (!rv.on || rv.ourSeek) return;
    var frame = Math.round(video.currentTime * (P.clips[cur] ? P.clips[cur].fps : 25));
    var idx = -1;
    for (var k = 0; k < rv.list.length; k++) {
      var y = rv.list[k];
      if (y.c > cur || (y.c === cur && y.e.frame >= frame)) { idx = k; break; }
    }
    card.hidden = true; $('[data-pr-reasons]').hidden = true;
    rv.waiting = false;
    if (idx < 0) { rv.target = null; status.textContent = 'No events after this point. Jump back, or Exit review.'; return; }
    rv.i = idx; rv.target = rv.list[idx];
    if (rv.target.c !== cur) { rvGo(idx); return; }
    var p = video.play(); if (p && p.catch) p.catch(function () {});
  });
  function vote(x, verdict, reason) {
    var e = x.e, clip = P.clips[x.c];
    return post(clip.review_url, {event_key: e.key, verdict: verdict, reason: reason || ''})
      .then(function (d) {
        if (e.mine === 'accepted') e.accepted = Math.max(0, (e.accepted || 0) - 1);
        if (e.mine === 'rejected') e.rejected = Math.max(0, (e.rejected || 0) - 1);
        e.mine = d.verdict;
        if (d.verdict === 'accepted') e.accepted = (e.accepted || 0) + 1;
        if (d.verdict === 'rejected') e.rejected = (e.rejected || 0) + 1;
        renderLanes();
      }).catch(function (err) { status.textContent = String(err.message || err); });
  }
  function rvGo(i) {
    if (i >= rv.list.length) { stopReview('Review finished: every event in this period has been shown.'); return; }
    rv.i = i; rv.waiting = false; rv.target = rv.list[i];
    card.hidden = true; $('[data-pr-reasons]').hidden = true;
    var x = rv.target, c = P.clips[x.c];
    rv.ourSeek = true;
    loadClip(x.c, Math.max(0, x.e.frame - Math.round(2 * c.fps)), true);
  }
  function rvArrive() {
    rv.waiting = true;          // first: pausing redraws, which calls back here
    video.pause();
    rv.ourSeek = true;
    ov.seekFrame(rv.target.e.frame);
    var e = rv.target.e;
    $('[data-pr-card-title]').textContent = e.action.toUpperCase() + ' tube ' + tube(e.reference) +
      ' · track ' + e.track_id + (e.at_edge ? ' · at edge' : '');
    $('[data-pr-card-where]').textContent = clock(e.clock).slice(11) + ' · event ' + (rv.i + 1) + ' of ' + rv.list.length;
    card.hidden = false;
  }
  function startEvents() {
    rv.list = all.filter(function (x) { return P.clips[x.c].can_review; });
    if (!rv.list.length) return;
    rv.on = true; trip.on = false;
    $('[data-pr-trip]') && ($('[data-pr-trip]').hidden = true);
    toggleButtons(true);
    rvGo(0);
  }
  function stopReview(msg) {
    rv.on = false; rv.waiting = false; trip.on = false; tp.on = false;
    if (ov) ov.focus(null);
    if (card) card.hidden = true;
    if ($('[data-pr-trip]')) $('[data-pr-trip]').hidden = true;
    video.pause();
    toggleButtons(false);
    if (msg) status.textContent = msg;
  }
  function toggleButtons(reviewing) {
    ['[data-pr-review-events]', '[data-pr-review-trips]'].forEach(function (s) { if ($(s)) $(s).hidden = reviewing; });
    if ($('[data-pr-exit]')) $('[data-pr-exit]').hidden = !reviewing;
  }

  // ── Trip review ──────────────────────────────────────────────────────────
  var trip = {on: false, i: 0, list: []};
  var sides = {};
  ['A', 'B'].forEach(function (k) {
    var box = $('[data-pr-side-' + k + ']');
    if (box) sides[k] = {box: box};
  });
  // Who left and who came back: the track, its species and marker, a few crops.
  function showSide(k, ref, e, chip) {
    var s = sides[k], c = P.clips[ref.clip];
    if (!s) return;
    var chipEl = $('[data-pr-trip-chip]', s.box);
    chipEl.textContent = chip + ' ' + tube(e.reference) + ' · ' + clock(e.clock).slice(11);
    chipEl.style.background = COLORS[e.action] || '#374151';
    $('[data-pr-trip-who]', s.box).textContent = 'Clip ' + clock(c.start).slice(11, 16) + ' · track ' + e.track_id +
      (e.species ? ' · ' + e.species + (e.species_confidence !== '' ? ' ' + Number(e.species_confidence).toFixed(2) : '') : '') +
      (e.marker ? ' · marker ' + e.marker : ' · no marker');
    var crops = $('[data-pr-trip-crops]', s.box);
    crops.innerHTML = '';
    fetch(c.crops_url.replace(/\/0\/crops\.json$/, '/' + e.track_id + '/crops.json'), {credentials: 'same-origin'})
      .then(function (r) { return r.ok ? r.json() : {crops: []}; })
      .then(function (d) {
        (d.crops || []).slice(0, 4).forEach(function (cr) {
          var img = document.createElement('img');
          img.src = cr.url; img.alt = 'track ' + e.track_id; img.loading = 'lazy';
          img.className = 'w-12 h-12 object-contain bg-gray-50 rounded border border-gray-200';
          crops.appendChild(img);
        });
      }).catch(function () {});
  }

  // ── Trip playback (memory/62): the trip plays in the one player above —
  // 2 s either side of the exit, then of the entry, or the whole trip between
  // them across clips (gaps with no recording are skipped).
  var PAD = 2;
  var tp = {segs: [], k: 0, on: false, a: null, b: null};
  function segsBetween(t0, t1, focus) {
    var out = [];
    P.clips.forEach(function (c, ci) {
      var lo = Math.max(t0, c.offset), hi = Math.min(t1, c.offset + (c.duration || 0));
      if (hi > lo) out.push({c: ci, from: Math.round((lo - c.offset) * c.fps),
                             to: Math.round((hi - c.offset) * c.fps), focus: focus});
    });
    return out;
  }
  function tripPlay(mode) {
    var a = tp.a, b = tp.b;
    if (!a || !b) return;
    var exitSegs = segsBetween(a.t - PAD, a.t + PAD, a.track_id);
    var entrySegs = segsBetween(b.t - PAD, b.t + PAD, b.track_id);
    tp.mode = mode;
    tp.segs = mode === 'exit' ? exitSegs : mode === 'entry' ? entrySegs
      : mode === 'whole' ? segsBetween(a.t - PAD, b.t + PAD, null) : exitSegs.concat(entrySegs);
    tp.k = 0; tp.on = true;
    tpStart();
  }
  var TP_LABEL = {exit: 'the exit', entry: 'the entry', whole: 'the whole trip', moments: 'the exit, then the entry'};
  function tpStart() {
    var sg = tp.segs[tp.k];
    var now = $('[data-pr-trip-now]');
    if (!sg) {
      tp.on = false; video.pause();
      if (now) now.textContent = 'Played ' + TP_LABEL[tp.mode] + '. Vote, or play it again.';
      return;
    }
    if (now) now.textContent = 'Playing ' + TP_LABEL[tp.mode] + ' · clip ' + clock(P.clips[sg.c].start).slice(11, 16);
    rv.ourSeek = true;
    loadClip(sg.c, sg.from, true).then(function () {
      if (ov) ov.focus(sg.focus != null && sg.focus !== '' ? Number(sg.focus) : null);
    });
  }
  function renderTripBar() {
    var bar = $('[data-pr-trip-bar]');
    if (!bar || !tp.a || !tp.b) return;
    var t0 = tp.a.t - PAD, t1 = tp.b.t + PAD, span = (t1 - t0) || 1;
    var at = function (t) { return Math.max(0, Math.min(100, 100 * (t - t0) / span)) + '%'; };
    bar.innerHTML = '';
    bar.style.background = 'repeating-linear-gradient(135deg, #f3f4f6 0 6px, #e5e7eb 6px 8px)';
    P.clips.forEach(function (c) {
      var lo = Math.max(t0, c.offset), hi = Math.min(t1, c.offset + (c.duration || 0));
      if (hi <= lo) return;
      var r = document.createElement('div');
      r.className = 'absolute top-0 h-full bg-gray-300 rounded-sm pointer-events-none';
      r.style.left = at(lo); r.style.width = (100 * (hi - lo) / span) + '%';
      bar.appendChild(r);
    });
    [[tp.a, 'exit'], [tp.b, 'enter']].forEach(function (p) {
      var m = document.createElement('div');
      m.className = 'absolute top-0 h-full w-1 rounded-sm pointer-events-none';
      m.style.left = at(p[0].t); m.style.background = COLORS[p[1]];
      bar.appendChild(m);
    });
    var head = document.createElement('div');
    head.setAttribute('data-pr-trip-head', '');
    head.className = 'absolute -top-0.5 -bottom-0.5 w-0.5 bg-gray-900 pointer-events-none';
    bar.appendChild(head);
    $('[data-pr-trip-bar-from]').textContent = clock(tp.a.clock).slice(11, 19) + ' exit';
    $('[data-pr-trip-bar-to]').textContent = 'entry ' + clock(tp.b.clock).slice(11, 19);
    bar.onclick = function (ev) {
      var r = bar.getBoundingClientRect();
      var t = t0 + span * Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
      for (var ci = 0; ci < P.clips.length; ci++) {
        var c = P.clips[ci];
        if (t >= c.offset && t < c.offset + (c.duration || 0)) {
          tp.on = false;
          rv.ourSeek = true;
          loadClip(ci, Math.round((t - c.offset) * c.fps), false);
          $('[data-pr-trip-now]').textContent = 'At ' + clock(c.start).slice(11, 16) + ' clip — press play to watch on.';
          return;
        }
      }
      $('[data-pr-trip-now]').textContent = 'No recording at that moment.';
    };
  }
  function tripHead() {
    var head = $('[data-pr-trip-head]');
    if (!head || !tp.a || !tp.b || cur < 0) return;
    var c = P.clips[cur], t0 = tp.a.t - PAD, span = (tp.b.t + PAD - t0) || 1;
    var t = c.offset + video.currentTime;
    head.style.left = Math.max(0, Math.min(100, 100 * (t - t0) / span)) + '%';
  }

  function tripGo(i) {
    var panel = $('[data-pr-trip]');
    if (i >= trip.list.length) { stopReview('Trip review finished: every suggested trip has been shown.'); return; }
    trip.i = i;
    var t = trip.list[i], a = findEvent(t.exit), b = findEvent(t.entry);
    $('[data-pr-relink]').hidden = true;
    $('[data-pr-trip-title]').textContent = 'Tube ' + tube(t.reference) + ' · out ' + clock(a.clock).slice(11) +
      ' → back ' + clock(b.clock).slice(11);
    $('[data-pr-trip-where]').textContent = Math.round(t.gap_sec / 60) + ' min out · trip ' + (i + 1) + ' of ' +
      trip.list.length + (t.same_marker ? ' · same marker' : '');
    showSide('A', t.exit, a, 'EXIT');
    showSide('B', t.entry, b, 'ENTER');
    panel.hidden = false;
    tp.a = a; tp.b = b;
    renderTripBar();
    tripPlay('moments');
  }
  function tripVote(t, entry, verdict, reason, relinked) {
    var exitClip = P.clips[t.exit.clip];
    return post(P.trip_url, {
      exit_job: exitClip.job, exit_run: exitClip.run, exit_key: t.exit.key,
      entry_job: entry ? P.clips[entry.clip].job : null, entry_run: entry ? P.clips[entry.clip].run : null,
      entry_key: entry ? entry.key : '',
      verdict: verdict, reason: reason || '', relinked: !!relinked
    }).then(function (d) {
      if (t.mine) t[t.mine.verdict] = Math.max(0, (t[t.mine.verdict] || 0) - 1);
      t.mine = d.verdict ? {verdict: d.verdict, entry_key: entry ? entry.key : ''} : null;
      if (d.verdict) t[d.verdict] = (t[d.verdict] || 0) + 1;
      renderLanes();
    }).catch(function (err) { status.textContent = String(err.message || err); });
  }
  function relinkChooser(t) {
    var a = findEvent(t.exit), list = $('[data-pr-relink-list]');
    list.innerHTML = '';
    all.filter(function (x) {
      return x.e.action === 'enter' && x.e.reference === t.reference && x.e.t > a.t && x.e.key !== t.entry.key;
    }).slice(0, 12).forEach(function (x) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'h-9 px-2.5 rounded-md border border-gray-300 bg-white hover:bg-gray-50';
      b.textContent = clock(x.e.clock).slice(11) + ' · track ' + x.e.track_id + (x.e.marker ? ' · ' + x.e.marker : '');
      b.addEventListener('click', function () {
        tripVote(t, {clip: x.c, key: x.e.key}, 'accepted', 'relinked', true).then(function () { tripGo(trip.i + 1); });
      });
      list.appendChild(b);
    });
    [['didn\'t come back', 'Didn\'t come back in this period'], ['can\'t tell', 'Can\'t tell']].forEach(function (r) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'h-9 px-2.5 rounded-md border border-gray-300 bg-white hover:bg-gray-50';
      b.textContent = r[1];
      b.addEventListener('click', function () {
        tripVote(t, r[0] === 'didn\'t come back' ? null : t.entry, 'rejected', r[0], false)
          .then(function () { tripGo(trip.i + 1); });
      });
      list.appendChild(b);
    });
    $('[data-pr-relink]').hidden = false;
  }
  function startTrips() {
    trip.list = P.trips.filter(function (t) { return P.clips[t.exit.clip].can_review; });
    if (!trip.list.length) { status.textContent = 'No suggested trips in this period.'; return; }
    trip.on = true; rv.on = false;
    if (card) card.hidden = true;
    toggleButtons(true);
    tripGo(0);
  }

  // ── Wiring ───────────────────────────────────────────────────────────────
  if ($('[data-pr-review-events]')) {
    $('[data-pr-review-events]').addEventListener('click', startEvents);
    $('[data-pr-review-trips]').addEventListener('click', startTrips);
    $('[data-pr-exit]').addEventListener('click', function () { stopReview(''); });
    $$('[data-pr-vote]').forEach(function (b) {
      b.addEventListener('click', function () {
        var verdict = b.getAttribute('data-pr-vote');
        vote(rv.target, verdict, '').then(function () {
          if (verdict === 'accepted') rvGo(rv.i + 1); else $('[data-pr-reasons]').hidden = false;
        });
      });
    });
    $$('[data-pr-reason]').forEach(function (b) {
      b.addEventListener('click', function () {
        vote(rv.target, 'rejected', b.getAttribute('data-pr-reason')).then(function () { rvGo(rv.i + 1); });
      });
    });
    $('[data-pr-next]').addEventListener('click', function () { rvGo(rv.i + 1); });
    $('[data-pr-replay]').addEventListener('click', function () { rvGo(rv.i); });
    $('[data-pr-skip]').addEventListener('click', function () { rvGo(rv.i + 1); });
    $('[data-pr-trip-yes]').addEventListener('click', function () {
      var t = trip.list[trip.i];
      tripVote(t, t.entry, 'accepted', '', false).then(function () { tripGo(trip.i + 1); });
    });
    $('[data-pr-trip-no]').addEventListener('click', function () { relinkChooser(trip.list[trip.i]); });
    $('[data-pr-trip-skip]').addEventListener('click', function () { tripGo(trip.i + 1); });
    $$('[data-pr-trip-play]').forEach(function (b) {
      b.addEventListener('click', function () { tripPlay(b.getAttribute('data-pr-trip-play')); });
    });
    document.addEventListener('keydown', function (ev) {
      if (!(rv.on || trip.on) || /^(INPUT|TEXTAREA|SELECT)$/.test((ev.target || {}).tagName || '')) return;
      var k = ev.key.toLowerCase(), hit = true;
      if (trip.on) {
        if (k === 'p') tripPlay('moments');
        else if (k === 'y') $('[data-pr-trip-yes]').click();
        else if (k === 'n') $('[data-pr-trip-no]').click();
        else if (ev.key === 'ArrowRight') tripGo(trip.i + 1);
        else if (ev.key === 'Escape') stopReview('');
        else hit = false;
      } else {
        if (k === 'y' && rv.waiting) $('[data-pr-vote="accepted"]').click();
        else if (k === 'n' && rv.waiting) $('[data-pr-vote="rejected"]').click();
        else if (k === 'r') rvGo(rv.i);
        else if (ev.key === 'ArrowRight') rvGo(rv.i + 1);
        else if (ev.key === 'Escape') stopReview('');
        else hit = false;
      }
      if (hit) { ev.preventDefault(); ev.stopPropagation(); }
    }, true);
  }

  // Add a missed event to the clip in the player (memory/62 §3b).
  var adder = (window.AddMissed && ov && $('[data-am-open]')) ? AddMissed({
    stage: $('[data-pr-stage]'), video: video, ov: ov, rotated: false,
    context: function () { return {url: P.clips[cur].add_url, csrf: csrf()}; },
    onSaved: function (d, saved) {
      var c = P.clips[cur];
      c.added = c.added || [];
      c.added.push({id: d.id, frame: saved.frame, end_frame: saved.end_frame, action: saved.action,
                    reference: d.reference, track_id: saved.track_id, by: 'you',
                    t: c.offset + saved.frame / (c.fps || 25)});
      status.textContent = 'Added to ' + c.title + '. It shows dashed on the lanes and in the Events CSV as review = added.';
      renderLanes();
    }
  }) : null;
  if (adder) {
    var canAdd = function () { return cur >= 0 && P.clips[cur] && P.clips[cur].can_review; };
    $('[data-am-open]').addEventListener('click', function () {
      if (!canAdd()) { status.textContent = 'You can\'t review this clip.'; return; }
      if (card) card.hidden = true;
      adder.open();
    });
    document.addEventListener('keydown', function (ev) {
      if (ev.key.toLowerCase() !== 'a' || adder.isOpen() || ev.metaKey || ev.ctrlKey || !canAdd() ||
          /^(INPUT|TEXTAREA|SELECT)$/.test((ev.target || {}).tagName || '')) return;
      ev.preventDefault();
      if (card) card.hidden = true;
      adder.open();
    });
  }

  renderTicks();
  renderStrip();
  renderLanes();
  if (P.clips.length) loadClip(0, 0, false);
  else status.textContent = 'No analysed clips with an Events step in this period.';
})();
