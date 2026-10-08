// Clip review, inline. Scanning a day's footage is watching one clip after
// another; a round trip to a detail page between each turns a minute of review
// into ten. Rows carry their own sources, so this needs no fetch.
(function () {
  var viewer = document.getElementById('clip-viewer');
  if (!viewer) return;
  var video  = document.getElementById('cv-video');
  var title  = document.getElementById('cv-title');
  var pos    = document.getElementById('cv-pos');
  var note   = document.getElementById('cv-note');
  var toggle = document.getElementById('cv-toggle');
  var resultsLink = document.getElementById('cv-results');
  var photoBox = document.getElementById('cv-photo');
  var photoHtml = {};
  var rows   = Array.prototype.slice.call(document.querySelectorAll('.clip-row[data-original]'));
  var at = -1;
  var showTracks = true;
  var stage = document.getElementById('cv-stage');
  var overlay = window.TrackOverlay ? TrackOverlay(video, document.getElementById('cv-canvas')) : null;
  var overlayUrl = null;

  // A photo row: the photo, its boxes and species, fetched once per photo.
  function renderPhoto(row, url) {
    video.pause();
    video.removeAttribute('src');
    stage.hidden = true;
    if (overlay) { overlay.clear(); overlayUrl = null; }
    toggle.hidden = true;
    photoBox.hidden = false;
    note.textContent = '';
    if (photoHtml[url]) { photoBox.innerHTML = photoHtml[url]; return; }
    photoBox.innerHTML = '<p class="text-sm text-gray-500 p-4">Loading the photo…</p>';
    fetch(url, {credentials: 'same-origin'})
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); })
      .then(function (html) {
        photoHtml[url] = html;
        if (rows[at] === row) photoBox.innerHTML = html;
      })
      .catch(function () {
        if (rows[at] === row) photoBox.innerHTML = '<p class="text-sm text-red-700 p-4">Could not load this photo.</p>';
      });
  }

  function render() {
    var row = rows[at];
    if (!row) return;
    var photoUrl = row.getAttribute('data-photo');
    if (photoUrl) {
      renderPhoto(row, photoUrl);
    } else {
      photoBox.hidden = true;
      stage.hidden = false;
      renderVideo(row);
    }

    title.textContent = row.getAttribute('data-title') || '';
    pos.textContent = (at + 1) + ' of ' + rows.length;

    var results = row.getAttribute('data-results');
    resultsLink.hidden = !results;
    if (results) resultsLink.setAttribute('href', results);

    rows.forEach(function (r) { r.classList.remove('bg-amber-50'); });
    row.classList.add('bg-amber-50');
  }

  function renderVideo(row) {
    var src = row.getAttribute('data-original');
    if (video.getAttribute('src') !== src) {
      video.setAttribute('src', src);
      video.load();
      var playing = video.play();
      if (playing && playing.catch) playing.catch(function () {});
    }
    // The clip's device may be mounted upside down: the video and the boxes
    // turn together, so the boxes stay on their insects.
    stage.classList.toggle('rotate-180', row.getAttribute('data-rotate') === '1');

    var url = row.getAttribute('data-overlay');
    toggle.hidden = !url || !overlay;
    toggle.textContent = showTracks ? 'Hide tracks' : 'Show tracks';
    if (!url || !overlay) {
      if (overlay) { overlay.clear(); overlayUrl = null; }
      note.textContent = 'This run produced no tracks, so this is the clip as recorded.';
      return;
    }
    if (!showTracks) {
      overlay.clear(); overlayUrl = null;
      note.textContent = 'The clip as recorded.';
      return;
    }
    if (overlayUrl === url) return;
    overlayUrl = url;
    note.textContent = 'Loading the tracks…';
    overlay.load(url)
      .then(function () {
        if (overlayUrl === url) note.textContent = 'Boxes and track ids drawn from this run as the clip plays · dashed = the tracker lost the insect.' + (resultsLink.hidden ? '' : ' Full results has the per-track view.');
      })
      .catch(function () {
        if (overlayUrl === url) note.textContent = 'Could not load the tracks; this is the clip as recorded.';
      });
  }

  function open(index) {
    if (index < 0 || index >= rows.length) return;
    at = index;
    viewer.hidden = false;
    render();
    viewer.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  }

  function close() {
    viewer.hidden = true;
    video.pause();
    video.removeAttribute('src');
    video.load();
    if (overlay) { overlay.clear(); overlayUrl = null; }
    rows.forEach(function (r) { r.classList.remove('bg-amber-50'); });
    at = -1;
  }

  // Rows are ordered newest-first, so "earlier" is further DOWN the list —
  // the same direction the review workspace uses. Going forward in the list
  // while the label says "earlier" is the bug that gets reported every time.
  function step(delta) { open(Math.min(Math.max(at + delta, 0), rows.length - 1)); }

  rows.forEach(function (row, i) {
    row.querySelectorAll('.cv-open').forEach(function (btn) {
      btn.addEventListener('click', function () { open(i); });
    });
  });
  document.getElementById('cv-prev').addEventListener('click', function () { step(1); });
  document.getElementById('cv-next').addEventListener('click', function () { step(-1); });
  document.getElementById('cv-close').addEventListener('click', close);
  toggle.addEventListener('click', function () { showTracks = !showTracks; render(); });

  document.addEventListener('keydown', function (e) {
    if (viewer.hidden) return;
    if (e.key === 'Escape') { close(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); step(1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); step(-1); }
  });
})();
