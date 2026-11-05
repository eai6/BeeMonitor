// Trip review (memory/62 §4b): every trip of one unit or site, one after
// another, in one player that moves from clip to clip.
//
// The page carries them as JSON (#period-data): clips (each with its offset,
// frame rate, overlay and stream URLs and events) and the trips.
(function () {
  'use strict';
  var root = document.getElementById('period-review');
  var dataEl = document.getElementById('period-data');
  if (!root || !dataEl) return;
  var P = JSON.parse(dataEl.textContent);
  var $ = function (sel, el) { return (el || root).querySelector(sel); };
  var $$ = function (sel, el) { return Array.prototype.slice.call((el || root).querySelectorAll(sel)); };
  var COLORS = {exit: '#0369a1', enter: '#b45309', stay: '#6d28d9'};

  var video = $('[data-pr-video]'), canvas = $('[data-pr-canvas]');
  var status = $('[data-pr-status]');
  var ov = window.TrackOverlay ? TrackOverlay(video, canvas) : null;
  // What the player draws, as on the clip page.
  $$('[data-pr-opt]').forEach(function (box) {
    box.addEventListener('change', function () {
      var o = {}; o[box.getAttribute('data-pr-opt')] = box.checked;
      if (ov) ov.set(o);
    });
  });
  var cur = -1;                 // index of the loaded clip
  var pendingFrame = null;      // frame to seek to once the clip has loaded

  function tube(ref) { return String(ref || '').replace(/^reference_/, ''); }
  function clock(iso) { return (iso || '').replace('T', ' ').slice(0, 19); }
  function csrf() { var el = $('[name=csrfmiddlewaretoken]'); return el ? el.value : ''; }
  function post(url, body) {
    return fetch(url, {method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf()},
      body: JSON.stringify(body)}).then(function (r) { return r.json(); })
      .then(function (d) { if (!d.ok) throw new Error(d.error || 'Could not save'); return d; });
  }

  // A trip's call, whoever made it: what review resumes by.
  var TRIP_CALL = {accepted: 'same bee', rejected: 'not this pair', disputed: 'reviewers disagree'};
  function tripCall(t) {
    if (t.mine) return t.mine.verdict;
    return t.accepted && t.rejected ? 'disputed' : t.accepted ? 'accepted' : t.rejected ? 'rejected' : '';
  }

  // Which trips are shown: the pipeline's, from every detected event, or those
  // of the reviewed events (rejected out, added in). A trip in both is one trip.
  var stage = 'before';
  try { if (localStorage.getItem('pr-stage') === 'after') stage = 'after'; } catch (e) {}
  function stageTrips() { return P.trips.filter(function (t) { return t[stage]; }); }
  // A trip's type (memory/78): a reviewer's, else the one from the tracks.
  var kind = 'all';
  function tripType(t) { return t.type_set || t.auto_type; }
  function shownTrips() {
    return stageTrips().filter(function (t) { return kind === 'all' || tripType(t) === kind; });
  }

  function findEvent(ref) {
    var c = P.clips[ref.clip];
    if (!c) return null;
    var same = function (e) { return e.key === ref.key; };
    return c.events.filter(same)[0] || (c.added || []).filter(same)[0];
  }

  function paintProgress() {
    var el = $('[data-pr-progress]');
    var shown = shownTrips(), tripsDone = shown.filter(tripCall).length;
    el.textContent = shown.length + ' trips · ' + tripsDone + ' reviewed · ' +
      (shown.length - tripsDone) + ' to go';
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
    if (trip.on || cur + 1 >= P.clips.length) return;
    loadClip(cur + 1, 0, true);
  });
  if (ov) ov.onFrame(function (info) {
    if (trip.on) tripHead();
    if (tp.on) {
      var sg = tp.segs[tp.k];
      if (sg && sg.c === cur && info.frame >= sg.to) { tp.k++; tpStart(); }
    }
  });

  // Playback speed in trip review. Set as the default too: a new clip's video
  // starts at the default, and a trip runs across clips.
  var SPEEDS = [0.25, 0.5, 1, 2, 4, 8, 16];
  var speed = 1;
  try { speed = Number(localStorage.getItem('pr-speed')) || 1; } catch (e) {}
  if (SPEEDS.indexOf(speed) < 0) speed = 1;
  function applySpeed(rate) {
    try { video.defaultPlaybackRate = rate; video.playbackRate = rate; } catch (e) {}
  }
  function paintSpeed() {
    $$('[data-pr-speed]').forEach(function (b) {
      var on = Number(b.getAttribute('data-pr-speed')) === speed;
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
      b.classList.toggle('bg-amber-700', on);
      b.classList.toggle('text-white', on);
      b.classList.toggle('bg-white', !on);
    });
  }
  var speedBox = $('[data-pr-speeds]');
  if (speedBox) {
    SPEEDS.forEach(function (rate) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'h-10 min-w-[44px] px-2 rounded border border-gray-300 bg-white hover:bg-gray-50 text-[13px]';
      b.setAttribute('data-pr-speed', rate);
      b.textContent = rate + '×';
      b.addEventListener('click', function () {
        speed = rate;
        try { localStorage.setItem('pr-speed', rate); } catch (e) {}
        applySpeed(rate);
        paintSpeed();
      });
      speedBox.appendChild(b);
    });
    paintSpeed();
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
    $('[data-pr-trip-who]', s.box).textContent = 'Clip ' + clock(c.start).slice(11, 16) +
      (e.by ? ' · added by ' + e.by : '') + (e.track_id ? ' · track ' + e.track_id : '') +
      (e.species ? ' · ' + e.species + (e.species_confidence !== '' ? ' ' + Number(e.species_confidence).toFixed(2) : '') : '') +
      (e.by ? '' : e.marker ? ' · marker ' + e.marker : ' · no marker');
    var crops = $('[data-pr-trip-crops]', s.box);
    crops.innerHTML = '';
    if (!e.track_id) return;
    fetch(c.crops_url.replace(/\/0\/crops\.json(?=\?|$)/, '/' + e.track_id + '/crops.json'), {credentials: 'same-origin'})
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
    bar.style.background = 'repeating-linear-gradient(135deg, var(--bm-stripe-a, #f3f4f6) 0 6px, var(--bm-stripe-b, #e5e7eb) 6px 8px)';
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

  function tripLength(sec) {
    if (sec < 60) return Math.round(sec) + ' s';
    var min = Math.round(sec / 60);
    return min < 60 ? min + ' min' : Math.floor(min / 60) + ' h ' + (min % 60) + ' min';
  }

  // ``still``: show the trip without playing it, as the page opens (a browser
  // won't start a video nobody asked for).
  function tripGo(i, still) {
    var panel = $('[data-pr-trip]');
    trip.i = i;
    var t = trip.list[i], a = findEvent(t.exit), b = findEvent(t.entry);
    $('[data-pr-relink]').hidden = true;
    $('[data-pr-trip-title]').textContent = 'Tube ' + tube(t.reference) + ' · out ' + clock(a.clock).slice(11) +
      ' → back ' + clock(b.clock).slice(11);
    $('[data-pr-trip-where]').textContent = tripLength(t.gap_sec) + ' out · trip ' + (i + 1) + ' of ' +
      trip.list.length + (t.same_marker ? ' · same marker' : '');
    var call = tripCall(t), callEl = $('[data-pr-trip-call]');
    callEl.hidden = !call;
    callEl.textContent = call ? 'Already reviewed: ' + TRIP_CALL[call] +
      (t.mine && t.mine.by ? ' (' + t.mine.by + ')' : '') + '.' +
      (t.mine && t.mine.reason ? ' Note: ' + t.mine.reason + '.' : '') + ' Vote again to change it.' : '';
    $('[data-pr-trip-first]').disabled = i === 0;
    $('[data-pr-trip-to]').value = i + 1;
    $('[data-pr-trip-to]').max = trip.list.length;
    $('[data-pr-trip-back]').disabled = i === 0;
    $('[data-pr-trip-next]').disabled = i + 1 >= trip.list.length;
    paintType(t);
    showSide('A', t.exit, a, 'EXIT');
    showSide('B', t.entry, b, 'ENTER');
    panel.hidden = false;
    tp.a = a; tp.b = b;
    renderTripBar();
    if (still) {
      var at = segsBetween(a.t - PAD, a.t + PAD, null)[0];
      tp.on = false;
      $('[data-pr-trip-now]').textContent = 'Press P to play the exit, then the entry.';
      if (at) loadClip(at.c, at.from, false);
    } else {
      tripPlay('moments');
    }
  }
  // After a vote, on to the next trip nobody has called, coming round to any
  // passed over.
  function tripNext() {
    var n = trip.list.length;
    for (var k = 1; k < n; k++) {
      var i = (trip.i + k) % n;
      if (!tripCall(trip.list[i])) { tripGo(i); return; }
    }
    tripGo(trip.i, true);
    $('[data-pr-trip-now]').textContent = 'Every suggested trip here has been reviewed.';
  }
  function tripVote(t, entry, verdict, reason, relinked) {
    var exitClip = P.clips[t.exit.clip];
    return post(P.trip_url, {
      exit_job: exitClip.job, exit_run: exitClip.run, exit_key: t.exit.key,
      entry_job: entry ? P.clips[entry.clip].job : null, entry_run: entry ? P.clips[entry.clip].run : null,
      entry_key: entry ? entry.key : '',
      verdict: verdict, reason: reason || '', relinked: !!relinked, before_review: !!t.before,
      trip_type: t.type_set || ''
    }).then(function (d) {
      // One call per trip (memory/74): this one replaces anyone else's.
      t.mine = d.verdict ? {verdict: d.verdict, entry_key: entry ? entry.key : '', reason: (reason || '').trim(),
                            by: 'you'} : null;
      t.accepted = d.verdict === 'accepted' ? 1 : 0;
      t.rejected = d.verdict === 'rejected' ? 1 : 0;
      paintProgress();
    }).catch(function (err) { status.textContent = String(err.message || err); });
  }
  // Not this pair: a note on why, in the reviewer's own words, then on.
  function askWhy(t) {
    var box = $('[data-pr-relink]'), note = $('[data-pr-trip-note]');
    note.value = (t.mine && t.mine.reason) || '';
    box.hidden = false;
    note.focus();
    box.onsubmit = function (ev) {
      ev.preventDefault();
      tripVote(t, t.entry, 'rejected', note.value, false).then(function () { tripNext(); });
    };
  }
  // This page is for reviewing trips: it opens on the first still to do, or
  // with none left, at the top to look them over.
  function startTrips() {
    trip.list = shownTrips().filter(function (t) { return P.clips[t.exit.clip].can_review; });
    if ($('[data-pr-trip]')) $('[data-pr-trip]').hidden = !trip.list.length;
    if (!$('[data-pr-trip]') || !trip.list.length) { trip.on = false; return false; }
    trip.on = true;
    applySpeed(speed);
    tripGo(Math.max(0, trip.list.findIndex(function (t) { return !tripCall(t); })), true);
    return true;
  }

  function paintButton(b, on) {
    b.setAttribute('aria-pressed', on ? 'true' : 'false');
    b.classList.toggle('bg-amber-700', on);
    b.classList.toggle('text-white', on);
    b.classList.toggle('bg-white', !on);
  }
  function paintKinds() {
    var list = stageTrips();
    $$('[data-pr-kind]').forEach(function (b) {
      var name = b.getAttribute('data-pr-kind');
      paintButton(b, name === kind);
      $('[data-pr-kind-n]', b).textContent = list.filter(function (t) {
        return name === 'all' || tripType(t) === name;
      }).length;
    });
  }
  function paintType(t) {
    $$('[data-pr-type]').forEach(function (b) { paintButton(b, b.getAttribute('data-pr-type') === tripType(t)); });
    $('[data-pr-type-why]').textContent = t.type_set ? 'Changed by a reviewer.' :
      t.type_reason ? 'Set automatically: ' + t.type_reason : 'No tube hop seen.';
  }
  // On a trip already called this saves now; otherwise with the Yes or No.
  function setType(t, type) {
    t.type_set = type === t.auto_type ? '' : type;
    paintType(t);
    paintKinds();
    if (!t.mine) return;
    var exitClip = P.clips[t.exit.clip];
    post(P.trip_url, {exit_job: exitClip.job, exit_run: exitClip.run, exit_key: t.exit.key,
                      entry_key: t.mine.entry_key, set_type: true, trip_type: t.type_set})
      .catch(function (err) { status.textContent = String(err.message || err); });
  }

  function paintStage() {
    $$('[data-pr-list]').forEach(function (b) {
      var name = b.getAttribute('data-pr-list');
      paintButton(b, name === stage);
      $('[data-pr-list-n]', b).textContent = P.trips.filter(function (t) { return t[name]; }).length;
    });
  }
  $$('[data-pr-list]').forEach(function (b) {
    b.addEventListener('click', function () {
      stage = b.getAttribute('data-pr-list');
      try { localStorage.setItem('pr-stage', stage); } catch (e) {}
      paintStage();
      showList();
    });
  });
  $$('[data-pr-kind]').forEach(function (b) {
    b.addEventListener('click', function () {
      kind = b.getAttribute('data-pr-kind');
      showList();
    });
  });
  function showList() {
    paintKinds();
    paintProgress();
    status.textContent = startTrips() ? '' : 'No trips in this list.';
  }

  // ── Wiring ───────────────────────────────────────────────────────────────
  if ($('[data-pr-trip]')) {
    $('[data-pr-trip-yes]').addEventListener('click', function () {
      var t = trip.list[trip.i];
      tripVote(t, t.entry, 'accepted', '', false).then(function () { tripNext(); });
    });
    $('[data-pr-trip-no]').addEventListener('click', function () { askWhy(trip.list[trip.i]); });
    $$('[data-pr-type]').forEach(function (b) {
      b.addEventListener('click', function () { setType(trip.list[trip.i], b.getAttribute('data-pr-type')); });
    });
    $('[data-pr-trip-next]').addEventListener('click', function () {
      if (trip.i + 1 < trip.list.length) tripGo(trip.i + 1);
    });
    $('[data-pr-trip-back]').addEventListener('click', function () { if (trip.i > 0) tripGo(trip.i - 1); });
    $('[data-pr-trip-first]').addEventListener('click', function () { tripGo(0); });
    $('[data-pr-trip-jump]').addEventListener('submit', function (ev) {
      ev.preventDefault();
      var to = $('[data-pr-trip-to]'), n = parseInt(to.value, 10);
      to.blur();
      if (n >= 1 && n <= trip.list.length) tripGo(n - 1);
      else to.value = trip.i + 1;
    });
    $$('[data-pr-trip-play]').forEach(function (b) {
      b.addEventListener('click', function () { tripPlay(b.getAttribute('data-pr-trip-play')); });
    });
    document.addEventListener('keydown', function (ev) {
      if (!trip.on || /^(INPUT|TEXTAREA|SELECT)$/.test((ev.target || {}).tagName || '')) return;
      var k = ev.key.toLowerCase(), hit = true;
      if (k === 'p') tripPlay('moments');
      else if (k === 'y') $('[data-pr-trip-yes]').click();
      else if (k === 'n') $('[data-pr-trip-no]').click();
      else if (ev.key === 'ArrowRight') $('[data-pr-trip-next]').click();
      else if (ev.key === 'ArrowLeft') $('[data-pr-trip-back]').click();
      else hit = false;
      if (hit) { ev.preventDefault(); ev.stopPropagation(); }
    }, true);
  }

  if ($('[data-pr-place]')) $('[data-pr-place]').addEventListener('change', function () {
    window.location.href = this.value;
  });
  paintStage();
  paintKinds();
  paintProgress();
  if (!P.clips.length) status.textContent = 'No analysed clips with an Events step in this period.';
  else if (!startTrips()) {
    loadClip(0, 0, false);
    status.textContent = 'No suggested trips in this period.';
  }
})();
