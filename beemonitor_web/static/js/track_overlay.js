// Tracks drawn over the original video, in step with playback (memory/46).
//
//   var ov = TrackOverlay(videoEl, canvasEl);
//   ov.load(url).then(...);          // the analysis:track_overlay payload
//   ov.set({species: false});        // boxes, species, trails, lost, regions
//   ov.focus(12);                    // follow one track; null for all
//   ov.onFrame(function (info) {});  // {frame, visible: [ids]} per drawn frame
//
// The canvas sits over the video element with the same box. Coordinates in the
// payload are the video's own pixels; they are scaled to where the picture is
// drawn inside the element (object-fit: contain letterboxes it).
(function (global) {
  'use strict';

  function color(id, alpha) {
    var hue = (id * 137.508) % 360;
    return 'hsla(' + hue.toFixed(1) + ',85%,58%,' + (alpha == null ? 1 : alpha) + ')';
  }

  function lowerBound(arr, value) {
    var lo = 0, hi = arr.length;
    while (lo < hi) { var mid = (lo + hi) >> 1; if (arr[mid] < value) lo = mid + 1; else hi = mid; }
    return lo;
  }

  function TrackOverlay(video, canvas) {
    var ctx = canvas.getContext('2d');
    var data = null;          // payload
    var byFrame = null;       // frame -> [start row, count] (Int32Array pairs)
    var perTrack = {};        // id -> {frames, cx, cy, lost}
    var opts = {boxes: true, species: true, trails: true, lost: true, regions: false};
    var focused = null;
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

    function label(id, lost, frame) {
      var text = String(id);
      var tr = data.tracks[id];
      if (opts.species && tr && tr.species) text += ' · ' + tr.species;
      if (lost) text += ' · lost ' + lostFor(id, frame).toFixed(1) + ' s';
      return text;
    }

    function drawRegions(L) {
      ctx.save();
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.font = '11px ui-monospace, monospace';
      (data.regions || []).forEach(function (r) {
        ctx.beginPath();
        r.points.forEach(function (p, i) {
          var x = L.ox + p[0] * L.vw * L.s, y = L.oy + p[1] * L.vh * L.s;
          if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();
        var p0 = r.points[0];
        ctx.fillText(r.label, L.ox + p0[0] * L.vw * L.s + 3, L.oy + p0[1] * L.vh * L.s + 12);
      });
      ctx.restore();
    }

    function drawTrail(id, frame, L, whole, alpha) {
      var t = perTrack[id]; if (!t) return;
      var end = lowerBound(t.frames, frame + 1);
      var start = whole ? 0 : lowerBound(t.frames, frame - Math.round(2 * fps()));
      if (end - start < 2) return;
      ctx.save();
      ctx.strokeStyle = color(id, alpha);
      ctx.lineWidth = whole ? 2.5 : 1.5;
      ctx.beginPath();
      for (var i = start; i < end; i++) {
        var x = L.ox + t.cx[i] * L.s, y = L.oy + t.cy[i] * L.s;
        // A gap of more than a second is two visits, not one path.
        if (i > start && t.frames[i] - t.frames[i - 1] > fps()) ctx.moveTo(x, y);
        else if (i === start) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    function draw(force) {
      if (!data) return;
      var frame = currentFrame();
      if (!force && frame === lastFrame) return;
      lastFrame = frame;
      var L = layout();
      sizeCanvas(L);
      ctx.clearRect(0, 0, L.ew, L.eh);
      if (opts.regions) drawRegions(L);

      var visible = [];
      var start = frame * 2 < byFrame.length ? byFrame[frame * 2] : -1;
      var count = start >= 0 ? byFrame[frame * 2 + 1] : 0;
      var rows = data.rows;
      ctx.font = '600 12px ui-monospace, SFMono-Regular, Menlo, monospace';
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
        var a = dim ? 0.25 : (lost ? 0.7 : 1);
        ctx.save();
        ctx.lineWidth = focused === id ? 3 : 2;
        ctx.strokeStyle = color(id, a);
        if (lost) ctx.setLineDash([5, 4]);
        ctx.strokeRect(x, y, w, h);
        ctx.restore();
        if (dim) continue;
        var text = label(id, lost, frame);
        var tw = ctx.measureText(text).width + 8, th = 16;
        var ty = y - th - 1 < 0 ? y + h + 1 : y - th - 1;
        ctx.fillStyle = color(id, lost ? 0.7 : 0.95);
        ctx.fillRect(x, ty, tw, th);
        ctx.fillStyle = '#111827';
        ctx.fillText(text, x + 4, ty + 2);
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

    var api = {
      load: function (url) {
        api.clear();
        return fetch(url, {credentials: 'same-origin'})
          .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
          .then(function (payload) { index(payload); loop(); redraw(); return payload; });
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
