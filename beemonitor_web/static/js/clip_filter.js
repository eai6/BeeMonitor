// Filtering a batch's clips in the page. A big batch is hundreds of rows, and
// finding "the clips from Unit B on the 8th with interactions" meant scrolling
// all of them. Every row is already on the page, so this filters and sorts in
// place: no request, and it keeps up as you type.
//
// Rows carry what they can be filtered on as data-f-* attributes; controls
// carry data-filter (q, unit, day, sort), data-filter-min (a row attribute to
// be at least) or data-filter-status. The filter lives in the address bar, so
// a filtered view survives the running batch's auto-refresh and can be sent on.
(function () {
  var bar = document.getElementById('clip-filter');
  if (!bar) return;
  var rows = Array.prototype.slice.call(document.querySelectorAll('.clip-row'));
  if (!rows.length) return;
  var parent = rows[0].parentNode;
  // The server's order (newest first), so "Newest" puts it back exactly.
  rows.forEach(function (r, i) { r.setAttribute('data-f-order', i); });
  // Sorted rows go back in before this, wherever they now are.
  var end = document.createComment('clip rows end');
  parent.insertBefore(end, rows[rows.length - 1].nextSibling);

  var controls = Array.prototype.slice.call(bar.querySelectorAll('[data-filter],[data-filter-min]'));
  var chips = Array.prototype.slice.call(document.querySelectorAll('[data-filter-status]'));
  var shownEl = document.getElementById('cf-shown');
  var clearEls = Array.prototype.slice.call(document.querySelectorAll('[data-filter-clear]'));
  var filteredOnly = Array.prototype.slice.call(document.querySelectorAll('[data-filter-active]'));
  var empty = document.getElementById('cf-empty');
  var status = 'all';

  function key(el) { return el.getAttribute('data-filter') || ('min_' + el.getAttribute('data-filter-min')); }
  function num(v) { return v === null || v === '' ? null : Number(v); }

  function readUrl() {
    var params = new URLSearchParams(window.location.search);
    controls.forEach(function (el) {
      var v = params.get(key(el));
      if (v === null) return;
      if (el.type === 'checkbox') el.checked = v === '1'; else el.value = v;
    });
    status = params.get('status') || 'all';
  }

  function writeUrl() {
    var params = new URLSearchParams(window.location.search);
    controls.forEach(function (el) {
      var v = el.type === 'checkbox' ? (el.checked ? '1' : '') : el.value.trim();
      if (el.getAttribute('data-filter') === 'sort' && v === 'new') v = '';
      if (v) params.set(key(el), v); else params.delete(key(el));
    });
    if (status !== 'all') params.set('status', status); else params.delete('status');
    var qs = params.toString();
    history.replaceState(null, '', window.location.pathname + (qs ? '?' + qs : '') + window.location.hash);
  }

  function matches(row) {
    if (status !== 'all' && row.getAttribute('data-f-status') !== status) return false;
    for (var i = 0; i < controls.length; i++) {
      var el = controls[i];
      var name = el.getAttribute('data-filter');
      var min = el.getAttribute('data-filter-min');
      if (min) {
        var floor = num(el.value.trim());
        if (floor === null) continue;
        var have = num(row.getAttribute('data-f-' + min));
        // No value means this pipeline never measured it: not "at least" anything.
        if (have === null || have < floor) return false;
      } else if (name === 'q') {
        var words = el.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
        var text = (row.getAttribute('data-f-text') || '').toLowerCase();
        for (var w = 0; w < words.length; w++) if (text.indexOf(words[w]) < 0) return false;
      } else if (name === 'unit' || name === 'day') {
        if (el.value && row.getAttribute('data-f-' + name) !== el.value) return false;
      } else if (name === 'empty') {
        if (el.checked && row.getAttribute('data-f-tracks') !== '0') return false;
      }
    }
    return true;
  }

  function sortRows(by) {
    var field = by === 'new' || by === 'old' || !by ? 'order' : by;
    var sign = by === 'old' ? -1 : 1;
    var sorted = rows.slice().sort(function (a, b) {
      var x = num(a.getAttribute('data-f-' + field));
      var y = num(b.getAttribute('data-f-' + field));
      if (field === 'order') return sign * (x - y);
      // Most first; rows with no value always last.
      if (x === null && y === null) return 0;
      if (x === null) return 1;
      if (y === null) return -1;
      return y - x;
    });
    sorted.forEach(function (r) { parent.insertBefore(r, end); });
  }

  // The "Run these clips through a pipeline" form follows the filter, so
  // "re-run the clips with no tracks" is: filter, then Run.
  function updateRerun(visible, filtered) {
    var ids = [];
    visible.forEach(function (r) {
      var v = r.getAttribute('data-f-video');
      if (v && ids.indexOf(v) < 0) ids.push(v);
    });
    document.querySelectorAll('[data-rerun-form]').forEach(function (form) {
      var box = form.querySelector('[data-rerun-ids]');
      if (!form._allIds) form._allIds = box.innerHTML;
      if (filtered) {
        box.innerHTML = '';
        ids.forEach(function (id) {
          var input = document.createElement('input');
          input.type = 'hidden'; input.name = 'video_ids'; input.value = id;
          box.appendChild(input);
        });
      } else {
        box.innerHTML = form._allIds;
      }
      var n = filtered ? ids.length : box.querySelectorAll('input').length;
      form.querySelectorAll('[data-rerun-count]').forEach(function (el) {
        el.textContent = n + (filtered ? ' filtered ' : ' ') + el.getAttribute('data-rerun-noun') + (n === 1 ? '' : 's');
      });
      var submit = form.querySelector('[type=submit]');
      if (submit) submit.disabled = n === 0;
    });
    document.querySelectorAll('[data-rerun-shown]').forEach(function (el) { el.textContent = ids.length; });
  }

  function apply() {
    var sortEl = bar.querySelector('[data-filter="sort"]');
    sortRows(sortEl ? sortEl.value : 'new');
    var visible = [];
    rows.forEach(function (r) {
      var ok = matches(r);
      r.style.display = ok ? '' : 'none';
      if (ok) visible.push(r);
    });
    var filtered = visible.length !== rows.length;
    if (shownEl) shownEl.textContent = visible.length;
    filteredOnly.forEach(function (el) { el.hidden = !filtered; });
    if (empty) empty.hidden = visible.length !== 0;
    chips.forEach(function (c) {
      var on = c.getAttribute('data-filter-status') === status;
      c.setAttribute('aria-pressed', on ? 'true' : 'false');
      c.classList.toggle('bg-amber-600', on);
      c.classList.toggle('text-white', on);
      c.classList.toggle('bg-gray-100', !on);
      c.classList.toggle('text-gray-700', !on);
    });
    var resultsOn = controls.filter(function (el) {
      return el.hasAttribute('data-filter-min') ? el.value.trim() !== '' : (el.getAttribute('data-filter') === 'empty' && el.checked);
    }).length;
    var badge = document.getElementById('cf-results-badge');
    if (badge) badge.textContent = resultsOn ? ' (' + resultsOn + ')' : '';
    updateRerun(visible, filtered);
    writeUrl();
  }

  function clear() {
    controls.forEach(function (el) {
      if (el.getAttribute('data-filter') === 'sort') return;
      if (el.type === 'checkbox') el.checked = false;
      else el.value = '';
    });
    status = 'all';
    apply();
  }

  controls.forEach(function (el) {
    el.addEventListener(el.tagName === 'SELECT' || el.type === 'checkbox' ? 'change' : 'input', apply);
  });
  chips.forEach(function (c) {
    c.addEventListener('click', function () { status = c.getAttribute('data-filter-status'); apply(); });
  });
  clearEls.forEach(function (el) { el.addEventListener('click', clear); });

  var more = document.getElementById('cf-more');
  var panel = document.getElementById('cf-results');
  if (more && panel) {
    more.addEventListener('click', function () {
      panel.hidden = !panel.hidden;
      more.setAttribute('aria-expanded', panel.hidden ? 'false' : 'true');
    });
  }

  readUrl();
  // Results filters in the link: show what is filtering the list.
  if (panel && controls.some(function (el) {
    return panel.contains(el) && (el.type === 'checkbox' ? el.checked : el.value.trim() !== '');
  })) { panel.hidden = false; if (more) more.setAttribute('aria-expanded', 'true'); }
  apply();
})();
