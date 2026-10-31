// Tracks drawn over the original video, in step with playback (memory/46).
//
//   var ov = TrackOverlay(videoEl, canvasEl);
//   ov.load(url).then(...);          // the analysis:track_overlay payload
//   ov.set({species: false});        // boxes, species, trails, lost, regions, events, interactions
//   ov.focus(12);                    // follow one track; null for all
//   ov.onFrame(function (info) {});  // {frame, visible: [ids]} per drawn frame
//
// The canvas sits over the video element with the same box. Coordinates in the
// payload are the video's own pixels; they are scaled to where the picture is
// drawn inside the element (object-fit: contain letterboxes it).
(function (global) {
  'use strict';

  // How things are drawn comes from the payload's ``style`` (apps/analysis/
  // overlay_spec.py), the spec the GPU renderer reads too (memory/61). These
  // are only what the page draws before it has loaded one.
  var STYLE = {
    hue_step: 137.508, sat: 85, light: 58, box_width: 2, dash: [5, 4], lost_alpha: 0.7,
    label_font_px: 12, label_height: 16, label_pad: 4, label_text_rgba: [17, 24, 39, 1],
    trail_seconds: 2, trail_gap_seconds: 1, trail_width: 1.5, trail_alpha: 0.6,
    region_width: 1, roi_rgba: [74, 222, 128, 0.8], reference_rgba: [255, 255, 255, 0.55],
    region_label_px: 11, event_seconds: 1.5,
    event_rgba: {exit: [3, 105, 161, 1], enter: [180, 83, 9, 1], stay: [109, 40, 217, 1]},
    glow_width: 3, chip_font_px: 12, chip_text_rgba: [255, 255, 255, 1],
    in_reference_width: 2.5, near_rgba: [253, 230, 138, 1], near_width: 2
  };

  function color(id, alpha) {
    var hue = (id * STYLE.hue_step) % 360;
    return 'hsla(' + hue.toFixed(1) + ',' + STYLE.sat + '%,' + STYLE.light + '%,' +
      (alpha == null ? 1 : alpha) + ')';
  }

  function rgba(c, alpha) {
    return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + (alpha == null ? c[3] : alpha) + ')';
  }

  function lowerBound(arr, value) {
    var lo = 0, hi = arr.length;
    while (lo < hi) { var mid = (lo + hi) >> 1; if (arr[mid] < value) lo = mid + 1; else hi = mid; }
    return lo;
  }

  // The full-window stage: the video fills it, letterboxed, and the canvas
  // follows the video's box as it always does.
  function installFullWindowStyle() {
    if (document.getElementById('ov-fs-style')) return;
    var st = document.createElement('style');
    st.id = 'ov-fs-style';
    st.textContent =
      '.ov-stage:fullscreen, .ov-stage:-webkit-full-screen, .ov-stage.ov-full {' +
      ' display:flex; align-items:center; justify-content:center; background:#000; width:100%; height:100%; }' +
      '.ov-stage.ov-full { position:fixed; inset:0; z-index:60; border-radius:0; }' +
      '.ov-stage:fullscreen video, .ov-stage:-webkit-full-screen video, .ov-stage.ov-full video {' +
      ' max-height:none !important; width:100%; height:100%; object-fit:contain; }' +
      '.ov-fs-btn { position:absolute; top:8px; right:8px; z-index:2; width:44px; height:44px;' +
      ' display:flex; align-items:center; justify-content:center; border-radius:6px;' +
      ' background:rgba(0,0,0,0.55); color:#fff; border:0; cursor:pointer; }' +
      '.ov-fs-btn:hover { background:rgba(0,0,0,0.75); }' +
      '.ov-fs-btn:focus-visible { outline:2px solid #fbbf24; outline-offset:2px; }';
    document.head.appendChild(st);
  }

  function TrackOverlay(video, canvas) {
    var ctx = canvas.getContext('2d');
    var data = null;          // payload
    var byFrame = null;       // frame -> [start row, count] (Int32Array pairs)
    var perTrack = {};        // id -> {frames, cx, cy, lost}
    var opts = {boxes: true, species: true, trails: true, lost: true, regions: false,
                events: true, interactions: true};
    var focused = null;
    var onlyRef = null;       // show one reference's events and spells only
    var listeners = [];
    var lastFrame = -1;
    var vfcHandle = null, rafHandle = null;
    var W = 7;

    function index(payload) {
      data = payload;
      W = payload.row_width || 7;
      var rows = payload.rows, n = rows.length / W;
      var maxFrame = n ? rows[(n - 1) * W] : 0;
      byFrame = new Int32Array((maxFrame + 1) * 2).fill(-1);
      perTrack = {};
      var tmp = {};
      for (var i = 0; i < n; i++) {
        var o = i * W, f = rows[o], id = rows[o + 1];
        if (byFrame[f * 2] < 0) { byFrame[f * 2] = i; byFrame[f * 2 + 1] = 0; }
        byFrame[f * 2 + 1]++;
        var t = tmp[id] || (tmp[id] = {frames: [], cx: [], cy: [], lost: []});
        t.frames.push(f);
        t.cx.push((rows[o + 2] + rows[o + 4]) / 2);
        t.cy.push((rows[o + 3] + rows[o + 5]) / 2);
        t.lost.push(rows[o + 6]);
      }
      Object.keys(tmp).forEach(function (id) {
        var t = tmp[id];
        perTrack[id] = {frames: Int32Array.from(t.frames), cx: Float32Array.from(t.cx),
                        cy: Float32Array.from(t.cy), lost: Uint8Array.from(t.lost)};
      });
    }

    function fps() { return (data && data.fps) || 25; }
    function currentFrame() { return Math.max(0, Math.round(video.currentTime * fps())); }

    // Where the picture sits inside the element, in CSS px.
    function layout() {
      var ew = video.clientWidth, eh = video.clientHeight;
      var vw = video.videoWidth || ew, vh = video.videoHeight || eh;
      var s = Math.min(ew / vw, eh / vh) || 1;
      return {ew: ew, eh: eh, vw: vw, vh: vh, s: s, ox: (ew - vw * s) / 2, oy: (eh - vh * s) / 2};
    }

    function sizeCanvas(L) {
      var dpr = global.devicePixelRatio || 1;
      var w = Math.round(L.ew * dpr), h = Math.round(L.eh * dpr);
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      canvas.style.width = L.ew + 'px';
      canvas.style.height = L.eh + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function lostFor(id, frame) {
      // Seconds since the track's last real detection.
      var t = perTrack[id]; if (!t) return 0;
      var i = lowerBound(t.frames, frame + 1) - 1;
      while (i >= 0 && t.lost[i]) i--;
      return i < 0 ? 0 : (frame - t.frames[i]) / fps();
    }

    // Grey: a track the group filter leaves out of the count (memory/65).
    var LEFT_OUT = [156, 163, 175, 1];
    function isLeftOut(id) {
      var g = (data.groups || {})[id];
      return !!(g && !g[1]);
    }

    function label(id, lost, frame) {
      var text = String(id);
      var tr = data.tracks[id];
      if (opts.species && tr && tr.species) text += ' · ' + tr.species;
      // Its insect group, when an Identify Species node grouped the tracks
      // (memory/65); a track the filter leaves out says so.
      var g = (data.groups || {})[id];
      if (g) text += ' · ' + g[0] + (g[1] ? '' : ' · not counted');
      if (lost) text += ' · lost ' + lostFor(id, frame).toFixed(1) + ' s';
      var inside = inRef[id];
      if (inside) text += ' · in ' + regionText(inside[3]) + ' · ' + ((frame - inside[0]) / fps()).toFixed(1) + ' s';
      return text;
    }

    // "reference 12" -> "12": on a hotel with 60 tubes the full names ran
    // into each other. Names the user typed ("hotel", "flower") stay whole.
    function regionText(label) {
      var m = /^(?:reference|nest|tube|region)[ _-]?(\d+)$/i.exec(label || '');
      return m ? m[1] : (label || '');
    }

    function drawRegions(L) {
      ctx.save();
      ctx.lineWidth = STYLE.region_width;
      ctx.setLineDash(STYLE.dash);
      ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.textBaseline = 'top';
      (data.regions || []).forEach(function (r) {
        var x0 = Infinity, y0 = Infinity, x1 = -Infinity;
        // The ROI (tracking area) green, reference objects white.
        ctx.strokeStyle = rgba(r.kind === 'roi' ? STYLE.roi_rgba : STYLE.reference_rgba);
        ctx.beginPath();
        r.points.forEach(function (p, i) {
          var x = L.ox + p[0] * L.vw * L.s, y = L.oy + p[1] * L.vh * L.s;
          x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x);
          if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();
        // The label sits inside the box's top-left corner and shrinks to fit
        // its width; one that still can't fit is left off rather than drawn
        // over its neighbours.
        var text = regionText(r.label), size = STYLE.region_label_px;
        if (!text) return;
        ctx.font = size + 'px ui-monospace, monospace';
        var w = ctx.measureText(text).width;
        while (w + 4 > x1 - x0 && size > 8) {
          size -= 1;
          ctx.font = size + 'px ui-monospace, monospace';
          w = ctx.measureText(text).width;
        }
        if (w + 4 > x1 - x0) return;
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(0,0,0,0.55)';
        ctx.fillRect(x0 + 1, y0 + 1, w + 4, size + 3);
        ctx.fillStyle = 'rgba(255,255,255,0.95)';
        ctx.fillText(text, x0 + 3, y0 + 2);
        ctx.setLineDash(STYLE.dash);
      });
      ctx.restore();
    }

    // ── Events and interactions (memory/61) ────────────────────────────────
    function regionPath(label, L) {
      var r = (data.regions || []).filter(function (g) { return g.label === label; })[0];
      if (!r) return null;
      ctx.beginPath();
      var x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      r.points.forEach(function (p, i) {
        var x = L.ox + p[0] * L.vw * L.s, y = L.oy + p[1] * L.vh * L.s;
        x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      });
      ctx.closePath();
      return {x0: x0, y0: y0, x1: x1, y1: y1};
    }

    function restroke(label, style, width, L) {
      ctx.save();
      var box = regionPath(label, L);
      if (box) { ctx.setLineDash([]); ctx.lineWidth = width; ctx.strokeStyle = style; ctx.stroke(); }
      ctx.restore();
      return box;
    }

    // The events whose chip shows at this frame: from an exit for
    // event_seconds, the event_seconds before an enter, a stay throughout.
    function activeEvents(frame) {
      var w = Math.round(STYLE.event_seconds * fps());
      return (data.events || []).filter(function (e) {
        if (onlyRef != null && e[3] !== onlyRef) return false;
        if (e[2] === 'exit') return frame >= e[0] && frame <= e[0] + w;
        if (e[2] === 'enter') return frame >= e[0] - w && frame <= e[0];
        if (e[2] === 'stay') return frame >= e[0] && frame <= (e[4] == null ? e[0] : e[4]);
        return false;
      });
    }

    function activeInteractions(frame) {
      return (data.interactions || []).filter(function (r) {
        if (onlyRef != null && !(r[4] === 'in_reference' && r[3] === onlyRef)) return false;
        return frame >= r[0] && frame <= r[1];
      });
    }

    // Where a track is at this frame: its last row at or before it.
    function posAt(id, frame) {
      var t = perTrack[id]; if (!t) return null;
      var i = lowerBound(t.frames, frame + 1) - 1;
      if (i < 0 || frame - t.frames[i] > fps()) return null;
      return {x: t.cx[i], y: t.cy[i]};
    }

    function chip(text, x, y, fill, textColor) {
      ctx.save();
      ctx.font = '700 ' + STYLE.chip_font_px + 'px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.textBaseline = 'top';
      var tw = ctx.measureText(text).width + 2 * STYLE.label_pad, th = STYLE.label_height + 2;
      var cx = Math.max(0, Math.min(x, canvas.clientWidth - tw));
      ctx.fillStyle = fill;
      ctx.fillRect(cx, y, tw, th);
      ctx.fillStyle = textColor;
      ctx.fillText(text, cx + STYLE.label_pad, y + 3);
      ctx.restore();
      return th;
    }

    function drawTrail(id, frame, L, whole, alpha) {
      var t = perTrack[id]; if (!t) return;
      var end = lowerBound(t.frames, frame + 1);
      var start = whole ? 0 : lowerBound(t.frames, frame - Math.round(STYLE.trail_seconds * fps()));
      if (end - start < 2) return;
      ctx.save();
      ctx.strokeStyle = color(id, alpha);
      ctx.lineWidth = whole ? 2.5 : STYLE.trail_width;
      ctx.beginPath();
      for (var i = start; i < end; i++) {
        var x = L.ox + t.cx[i] * L.s, y = L.oy + t.cy[i] * L.s;
        // A gap of more than a second is two visits, not one path.
        if (i > start && t.frames[i] - t.frames[i - 1] > STYLE.trail_gap_seconds * fps()) ctx.moveTo(x, y);
        else if (i === start) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    var inRef = {};   // track id -> its active in-reference interaction, this frame

    function draw(force) {
      if (!data) return;
      var frame = currentFrame();
      if (!force && frame === lastFrame) return;
      lastFrame = frame;
      var L = layout();
      sizeCanvas(L);
      ctx.clearRect(0, 0, L.ew, L.eh);
      if (opts.regions) drawRegions(L);

      // Interactions and events under the boxes: reference re-strokes.
      inRef = {};
      var spells = opts.interactions ? activeInteractions(frame) : [];
      spells.forEach(function (r) {
        if (r[4] !== 'in_reference') return;
        inRef[r[2]] = r;
        restroke(r[3], color(r[2]), STYLE.in_reference_width, L);
      });
      var evs = opts.events ? activeEvents(frame) : [];
      evs.forEach(function (e) {
        restroke(e[3], rgba(STYLE.event_rgba[e[2]] || STYLE.event_rgba.exit), STYLE.glow_width, L);
      });
      var boxes = {};

      var visible = [];
      var start = frame * 2 < byFrame.length ? byFrame[frame * 2] : -1;
      var count = start >= 0 ? byFrame[frame * 2 + 1] : 0;
      var rows = data.rows;
      ctx.font = '600 ' + STYLE.label_font_px + 'px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.textBaseline = 'top';
      for (var k = 0; k < count; k++) {
        var o = (start + k) * W;
        var id = rows[o + 1], lost = rows[o + 6];
        if (lost && !opts.lost) continue;
        visible.push({id: id, lost: !!lost});
        var dim = focused != null && focused !== id;
        if (opts.trails || focused === id) drawTrail(id, frame, L, focused === id, dim ? 0.15 : 0.6);
        if (!opts.boxes && focused !== id) continue;
        var x = L.ox + rows[o + 2] * L.s, y = L.oy + rows[o + 3] * L.s;
        var w = (rows[o + 4] - rows[o + 2]) * L.s, h = (rows[o + 5] - rows[o + 3]) * L.s;
        var a = dim ? 0.25 : (lost ? STYLE.lost_alpha : 1);
        var gone = isLeftOut(id);
        ctx.save();
        ctx.strokeStyle = gone ? rgba(LEFT_OUT, a) : color(id, a);
        if (lost) ctx.setLineDash(STYLE.dash);
        ctx.lineWidth = focused === id ? 3 : STYLE.box_width;
        ctx.strokeRect(x, y, w, h);
        ctx.restore();
        boxes[id] = {x: x, y: y, w: w, h: h};
        if (dim) continue;
        var text = label(id, lost, frame);
        var tw = ctx.measureText(text).width + 2 * STYLE.label_pad, th = STYLE.label_height;
        var ty = y - th - 1 < 0 ? y + h + 1 : y - th - 1;
        ctx.fillStyle = gone ? rgba(LEFT_OUT, lost ? STYLE.lost_alpha : 0.95)
                             : color(id, lost ? STYLE.lost_alpha : 0.95);
        ctx.fillRect(x, ty, tw, th);
        ctx.fillStyle = rgba(STYLE.label_text_rgba);
        ctx.fillText(text, x + STYLE.label_pad, ty + 2);
      }

      // Near: a line between the two insects, labelled with the distance.
      spells.forEach(function (r) {
        if (r[4] !== 'near') return;
        var a = posAt(r[2], frame), b = posAt(r[3], frame);
        if (!a || !b) return;
        var ax = L.ox + a.x * L.s, ay = L.oy + a.y * L.s, bx = L.ox + b.x * L.s, by = L.oy + b.y * L.s;
        ctx.save();
        ctx.strokeStyle = rgba(STYLE.near_rgba);
        ctx.lineWidth = STYLE.near_width;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
        ctx.restore();
        var d = r[5] == null ? '' : ' ' + Number(r[5]).toFixed(1) + ' bl';
        chip('near' + d, (ax + bx) / 2 - 24, (ay + by) / 2 + 4, 'rgba(0,0,0,0.65)', rgba(STYLE.near_rgba));
      });

      // Event chips: under the insect's box, or at the reference when the
      // insect is not in this frame (after an entry, before an exit).
      var stacked = {};
      evs.forEach(function (e) {
        var text = e[2].toUpperCase() + ' ' + regionText(e[3]) + (e[5] ? ' · at edge' : '');
        var b = boxes[e[1]], x, y;
        if (b) { x = b.x; y = b.y + b.h + STYLE.label_height + 4; }
        else {
          var box = regionPath(e[3], L);
          if (!box) return;
          x = box.x0; y = box.y1 + 2;
        }
        var key = Math.round(x) + ':' + Math.round(y);
        y += (stacked[key] || 0);
        stacked[key] = (stacked[key] || 0) + chip(text, x, y,
          rgba(STYLE.event_rgba[e[2]] || STYLE.event_rgba.exit), rgba(STYLE.chip_text_rgba)) + 2;
      });

      // Events reviewers added as missed (memory/62 §3b): dashed chips, so
      // what people added never looks like what the system found.
      if (opts.events) {
        var w = Math.round(STYLE.event_seconds * fps());
        (data.added || []).forEach(function (a) {
          var f = a[0], on = a[2] === 'exit' ? frame >= f && frame <= f + w
            : a[2] === 'enter' ? frame >= f - w && frame <= f
            : frame >= f && frame <= (a[4] == null ? f : a[4]);
          if (!on || (onlyRef != null && a[3] !== onlyRef)) return;
          var b = a[1] != null ? boxes[a[1]] : null, x, y;
          if (b) { x = b.x; y = b.y + b.h + STYLE.label_height + 4; }
          else {
            var box = regionPath(a[3], L);
            if (!box) return;
            x = box.x0; y = box.y1 + 2;
          }
          var key = Math.round(x) + ':' + Math.round(y);
          y += (stacked[key] || 0);
          var col = rgba(STYLE.event_rgba[a[2]] || STYLE.event_rgba.exit);
          ctx.save();
          ctx.font = '700 ' + STYLE.chip_font_px + 'px ui-monospace, SFMono-Regular, Menlo, monospace';
          ctx.textBaseline = 'top';
          var text = '+ ' + a[2].toUpperCase() + ' ' + regionText(a[3]) + ' (added)';
          var tw = ctx.measureText(text).width + 2 * STYLE.label_pad, th = STYLE.label_height + 2;
          ctx.fillStyle = 'rgba(255,255,255,0.92)';
          ctx.fillRect(x, y, tw, th);
          ctx.setLineDash(STYLE.dash); ctx.lineWidth = 2; ctx.strokeStyle = col;
          ctx.strokeRect(x, y, tw, th);
          ctx.fillStyle = col;
          ctx.fillText(text, x + STYLE.label_pad, y + 3);
          ctx.restore();
          stacked[key] = (stacked[key] || 0) + th + 2;
        });
      }

      listeners.forEach(function (fn) { fn({frame: frame, visible: visible}); });
    }

    function loop() {
      if (video.requestVideoFrameCallback) {
        vfcHandle = video.requestVideoFrameCallback(function () { draw(false); loop(); });
      } else {
        rafHandle = requestAnimationFrame(function () { draw(false); loop(); });
      }
    }
    function stopLoop() {
      if (vfcHandle != null && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(vfcHandle);
      if (rafHandle != null) cancelAnimationFrame(rafHandle);
      vfcHandle = rafHandle = null;
    }

    function redraw() { draw(true); }
    ['seeked', 'loadedmetadata', 'loadeddata', 'pause'].forEach(function (ev) {
      video.addEventListener(ev, redraw);
    });
    var ro = global.ResizeObserver ? new ResizeObserver(redraw) : null;
    if (ro) ro.observe(video); else global.addEventListener('resize', redraw);

    // Full window. The browser's own fullscreen takes the <video> alone and
    // leaves the canvas behind, so the overlay vanished. The stage — video and
    // canvas together — goes full window instead, from our button or a double
    // click; the video's own fullscreen control is turned off.
    var stage = canvas.parentElement;
    installFullWindowStyle();
    stage.classList.add('ov-stage');
    if (video.controlsList && video.controlsList.add) video.controlsList.add('nofullscreen');
    video.setAttribute('controlslist', 'nofullscreen');
    var fsButton = document.createElement('button');
    fsButton.type = 'button';
    fsButton.className = 'ov-fs-btn';
    fsButton.setAttribute('aria-label', 'Full window');
    fsButton.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
      'stroke-width="2" aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>';
    stage.appendChild(fsButton);
    function isFull() {
      return document.fullscreenElement === stage || document.webkitFullscreenElement === stage ||
             stage.classList.contains('ov-full');
    }
    function toggleFull() {
      if (isFull()) {
        if (document.fullscreenElement || document.webkitFullscreenElement) {
          (document.exitFullscreen || document.webkitExitFullscreen).call(document);
        }
        stage.classList.remove('ov-full');
      } else if (stage.requestFullscreen || stage.webkitRequestFullscreen) {
        (stage.requestFullscreen || stage.webkitRequestFullscreen).call(stage);
      } else {
        stage.classList.add('ov-full');   // no Fullscreen API (iPhone): fill the window
      }
      setTimeout(redraw, 50);
    }
    fsButton.addEventListener('click', function (e) { e.stopPropagation(); toggleFull(); });
    video.addEventListener('dblclick', function (e) { e.preventDefault(); toggleFull(); });
    ['fullscreenchange', 'webkitfullscreenchange'].forEach(function (ev) {
      document.addEventListener(ev, function () {
        fsButton.setAttribute('aria-label', isFull() ? 'Exit full window' : 'Full window');
        setTimeout(redraw, 50);
      });
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && stage.classList.contains('ov-full')) toggleFull();
    });

    var api = {
      load: function (url) {
        api.clear();
        // The boxes are a stored file; the events and the references they were
        // counted against are fetched live, so a recount shows at once.
        var live = url + (url.indexOf('?') < 0 ? '?' : '&') + 'part=events';
        var get = function (u) {
          return fetch(u, {credentials: 'same-origin'})
            .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); });
        };
        return Promise.all([get(url), get(live).catch(function () { return {}; })])
          .then(function (parts) {
            var payload = parts[0];
            payload.events = parts[1].events || [];
            payload.interactions = parts[1].interactions || [];
            payload.regions = parts[1].regions || [];
            payload.added = parts[1].added || [];
            payload.groups = parts[1].groups || {};
            payload.group_options = parts[1].group_options || [];
            payload.can_regroup = !!parts[1].can_regroup;
            if (parts[1].style) STYLE = parts[1].style;
            index(payload); loop(); redraw(); return payload;
          });
      },
      clear: function () {
        stopLoop();
        data = null; byFrame = null; perTrack = {}; focused = null; lastFrame = -1;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      },
      set: function (o) { Object.keys(o).forEach(function (k) { opts[k] = !!o[k]; }); redraw(); },
      options: function () { return Object.assign({}, opts); },
      focus: function (id) { focused = id == null ? null : Number(id); redraw(); },
      // One reference only (the event lanes' tube names): its events and the
      // spells inside it; null for all.
      onlyReference: function (ref) { onlyRef = ref == null ? null : String(ref); redraw(); },
      focused: function () { return focused; },
      onFrame: function (fn) { listeners.push(fn); },
      data: function () { return data; },
      fps: fps,
      color: color,
      frame: currentFrame,
      // [[startFrame, endFrame], ...]: where a track is on screen, gaps over 1 s split it.
      spans: function (id) {
        var t = perTrack[id]; if (!t || !t.frames.length) return [];
        var out = [[t.frames[0], t.frames[0]]];
        for (var i = 1; i < t.frames.length; i++) {
          var last = out[out.length - 1];
          if (t.frames[i] - last[1] > fps()) out.push([t.frames[i], t.frames[i]]);
          else last[1] = t.frames[i];
        }
        return out;
      },
      // The box under a point in the canvas (CSS px), for click-to-follow.
      hit: function (px, py) {
        if (!data) return null;
        var L = layout(), frame = currentFrame();
        var start = frame * 2 < byFrame.length ? byFrame[frame * 2] : -1;
        if (start < 0) return null;
        var rows = data.rows, best = null, bestArea = Infinity;
        for (var k = 0; k < byFrame[frame * 2 + 1]; k++) {
          var o = (start + k) * W;
          if (rows[o + 6] && !opts.lost) continue;
          var x = L.ox + rows[o + 2] * L.s, y = L.oy + rows[o + 3] * L.s;
          var w = (rows[o + 4] - rows[o + 2]) * L.s, h = (rows[o + 5] - rows[o + 3]) * L.s;
          if (px >= x - 4 && px <= x + w + 4 && py >= y - 4 && py <= y + h + 4 && w * h < bestArea) {
            best = rows[o + 1]; bestArea = w * h;
          }
        }
        return best;
      },
      seekFrame: function (f) { video.currentTime = Math.max(0, f) / fps() + 0.0001; },
      step: function (delta) { video.pause(); api.seekFrame(currentFrame() + delta); },
    };
    return api;
  }

  global.TrackOverlay = TrackOverlay;
})(window);
