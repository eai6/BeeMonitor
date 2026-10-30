// Add an event the system missed (memory/62 §3b), shared by the clip page and
// period review.
//
//   var adder = AddMissed({stage, video, ov, rotated, context, onSaved, status});
//   adder.open();       // pause on the current frame and show the card
//
// context() returns {url, csrf} for the clip now in the player. The card sits
// inside the stage, so it stays in full window. While it is open, clicking a
// reference object on the video picks it as the tube.
(function (global) {
  'use strict';

  var ACTIONS = ['exit', 'enter', 'stay'];

  function inPolygon(x, y, pts) {
    var inside = false;
    for (var i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      var xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function AddMissed(o) {
    var stage = o.stage, video = o.video, ov = o.ov;
    var visible = [];
    if (ov) ov.onFrame(function (info) { visible = info.visible || []; });

    var card = el('div', 'absolute left-1/2 bottom-12 -translate-x-1/2 w-[min(600px,94%)] rounded-lg ' +
      'bg-gray-900/95 text-white p-3.5 flex flex-col gap-2.5 z-20 text-[13px]' + (o.rotated ? ' rotate-180' : ''));
    card.hidden = true;
    card.setAttribute('role', 'dialog');
    card.setAttribute('aria-label', 'Add an event the system missed');
    stage.appendChild(card);

    var st = null;   // {frame, action, reference, track, end, note}

    function fmt(frame) {
      var sec = frame / ((ov && ov.fps()) || 25), m = Math.floor(sec / 60), s = sec - m * 60;
      return String(m).padStart(2, '0') + ':' + s.toFixed(2).padStart(5, '0');
    }
    function tubeName(ref) { return String(ref || '').replace(/^reference_/, ''); }

    function chip(label, on, onClick) {
      var b = el('button', 'h-8 px-3 rounded-full border ' +
        (on ? 'bg-white text-gray-900 border-white font-bold' : 'border-gray-400 hover:bg-gray-800'), label);
      b.type = 'button';
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
      b.addEventListener('click', onClick);
      return b;
    }

    function render(message, duplicate) {
      card.innerHTML = '';
      var head = el('div', 'flex items-baseline justify-between gap-3');
      head.appendChild(el('strong', 'text-[15px]', 'Add a missed event at ' + fmt(st.frame)));
      head.appendChild(el('span', 'font-mono text-xs text-gray-300', 'frame ' + st.frame));
      card.appendChild(head);

      var row = el('div', 'flex flex-wrap items-center gap-2');
      row.appendChild(el('span', 'w-14 text-gray-300', 'What'));
      ACTIONS.forEach(function (a) {
        row.appendChild(chip(a, st.action === a, function () { st.action = a; render(); }));
      });
      card.appendChild(row);

      row = el('div', 'flex flex-wrap items-center gap-2');
      row.appendChild(el('span', 'w-14 text-gray-300', 'Tube'));
      row.appendChild(el('strong', '', st.reference ? 'tube ' + tubeName(st.reference) : 'none yet'));
      row.appendChild(el('span', 'text-gray-300', 'Click a tube on the video, or type its number:'));
      var input = el('input', 'w-16 h-8 px-2 rounded border border-gray-500 bg-gray-950 text-white');
      input.type = 'text'; input.inputMode = 'numeric';
      input.setAttribute('aria-label', 'Tube number');
      input.value = tubeName(st.reference);
      input.addEventListener('change', function () { st.reference = input.value ? 'reference_' + input.value.trim() : ''; render(); });
      row.appendChild(input);
      card.appendChild(row);

      row = el('div', 'flex flex-wrap items-center gap-2');
      row.appendChild(el('span', 'w-14 text-gray-300', 'Insect'));
      visible.filter(function (v) { return !v.lost; }).slice(0, 6).forEach(function (v) {
        row.appendChild(chip('track ' + v.id, st.track === String(v.id), function () { st.track = String(v.id); render(); }));
      });
      row.appendChild(chip('Not tracked', st.track === '', function () { st.track = ''; render(); }));
      card.appendChild(row);

      if (st.action === 'stay') {
        row = el('div', 'flex flex-wrap items-center gap-2');
        row.appendChild(el('span', 'w-14 text-gray-300', 'Until'));
        row.appendChild(el('span', 'font-mono', st.end != null ? fmt(st.end) : 'play on, then'));
        var endBtn = el('button', 'h-8 px-3 rounded-full border border-gray-400 hover:bg-gray-800', 'End here');
        endBtn.type = 'button';
        endBtn.addEventListener('click', function () { video.pause(); st.end = ov ? ov.frame() : st.frame; render(); });
        row.appendChild(endBtn);
        card.appendChild(row);
      }

      var note = el('input', 'h-8 px-2 rounded border border-gray-500 bg-gray-950 text-white w-full');
      note.type = 'text'; note.placeholder = 'Note (optional)';
      note.setAttribute('aria-label', 'Note');
      note.value = st.note || '';
      note.addEventListener('input', function () { st.note = note.value; });
      card.appendChild(note);

      if (message) card.appendChild(el('p', duplicate ? 'm-0 text-amber-300' : 'm-0 text-red-300', message));

      var actions = el('div', 'flex flex-wrap justify-end gap-2');
      var cancel = el('button', 'h-11 px-3.5 rounded border border-gray-500 hover:bg-gray-800', 'Cancel (Esc)');
      cancel.type = 'button';
      cancel.addEventListener('click', close);
      actions.appendChild(cancel);
      var save = el('button', 'h-11 px-5 rounded bg-amber-700 hover:bg-amber-800 font-bold',
        duplicate ? 'Add anyway' : 'Add event');
      save.type = 'button';
      save.addEventListener('click', function () { submit(!!duplicate); });
      actions.appendChild(save);
      card.appendChild(actions);
    }

    function submit(force) {
      if (!st.reference) { render('Pick the tube first.'); return; }
      if (st.action === 'stay' && st.end == null) { render('Play on and press End here to mark when the stay ended.'); return; }
      var ctx = o.context();
      fetch(ctx.url, {
        method: 'POST', credentials: 'same-origin',
        headers: {'Content-Type': 'application/json', 'X-CSRFToken': ctx.csrf},
        body: JSON.stringify({frame: st.frame, action: st.action, reference: st.reference,
                              track_id: st.track || '', end_frame: st.end, note: st.note || '', force: force})
      }).then(function (r) { return r.json(); }).then(function (d) {
        if (d.duplicate) {
          render('The system already has ' + st.action + ' tube ' + tubeName(st.reference) + ' at ' +
                 fmt(d.near.frame) + (d.near.track_id !== '' ? ' (track ' + d.near.track_id + ')' : '') +
                 '. Is this a different event?', true);
          return;
        }
        if (!d.ok) { render(d.error || 'Could not add it.'); return; }
        var saved = {frame: st.frame, end_frame: st.end, action: st.action, reference: st.reference,
                     track_id: st.track || ''};
        close();
        if (o.onSaved) o.onSaved(d, saved);
      }).catch(function () { render('Could not add it.'); });
    }

    // While the card is open a click on the video picks a reference object.
    stage.addEventListener('click', function (ev) {
      if (!st || card.contains(ev.target) || !ov || !ov.data()) return;
      var r = video.getBoundingClientRect();
      var x = ev.clientX - r.left, y = ev.clientY - r.top;
      if (o.rotated) { x = r.width - x; y = r.height - y; }
      var vw = video.videoWidth || r.width, vh = video.videoHeight || r.height;
      var s = Math.min(r.width / vw, r.height / vh) || 1;
      var nx = (x - (r.width - vw * s) / 2) / (vw * s), ny = (y - (r.height - vh * s) / 2) / (vh * s);
      var hit = (ov.data().regions || []).filter(function (g) {
        return g.kind !== 'roi' && inPolygon(nx, ny, g.points);
      })[0];
      ev.preventDefault(); ev.stopPropagation();
      if (hit) { st.reference = hit.label; render(); }
    }, true);

    document.addEventListener('keydown', function (ev) {
      if (st && ev.key === 'Escape') { ev.preventDefault(); ev.stopPropagation(); close(); }
    }, true);

    function open() {
      video.pause();
      st = {frame: ov ? ov.frame() : 0, action: 'enter', reference: '', track: null, end: null, note: ''};
      if (ov) ov.set({regions: true});
      render();
      card.hidden = false;
    }
    function close() { st = null; card.hidden = true; }

    return {open: open, close: close, isOpen: function () { return !!st; }};
  }

  global.AddMissed = AddMissed;
})(window);
