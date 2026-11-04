// Filtering a batch's clips in the page. A big batch is hundreds of rows, and
// finding "the clips from Unit B on the 8th with interactions" meant scrolling
// all of them. Every row is already on the page, so this filters and sorts in
// place: no request, and it keeps up as you type.
//
// Rows carry what they can be filtered on as data-f-* attributes; controls
// carry data-filter (q, unit, site, day, reviewed, by, sort), data-filter-min (a row attribute to
// be at least) or data-filter-status. The filter lives in the address bar, so
// a filtered view survives the running batch's auto-refresh and can be sent on.
//
// A big batch (memory/77) can't put every row in the page. It sends #clip-index,
// each clip's data-f-* values, and rows here are stand-ins for those: filtering,
// sorting and everything that follows the filter work on them as on real rows,
// and only the page of clips on screen is fetched as rows.
(function () {
  var bar = document.getElementById('clip-filter');
  if (!bar) return;
  var indexEl = document.getElementById('clip-index');
  var pager = document.getElementById('clip-pager');
  function standIn(attrs) {
    return {style: {},
            getAttribute: function (n) {
              var k = n.replace('data-f-', '');
              return k in attrs ? String(attrs[k]) : null;
            },
            setAttribute: function (n, v) { attrs[n.replace('data-f-', '')] = v; }};
  }
  var rows = indexEl ? JSON.parse(indexEl.textContent).map(standIn)
                     : Array.prototype.slice.call(document.querySelectorAll('.clip-row'));
  if (!rows.length) return;
  if (indexEl) window.bmClipRows = rows;
  var parent = indexEl ? null : rows[0].parentNode;
  // The server's order (newest first), so "Newest" puts it back exactly.
  rows.forEach(function (r, i) { r.setAttribute('data-f-order', i); });
  // Sorted rows go back in before this, wherever they now are.
  var end = null;
  if (parent) {
    end = document.createComment('clip rows end');
    parent.insertBefore(end, rows[rows.length - 1].nextSibling);
  }
  var ordered = rows, page = 1;

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
    page = Math.max(1, parseInt(params.get('page'), 10) || 1);
  }

  function writeUrl() {
    var params = new URLSearchParams(window.location.search);
    controls.forEach(function (el) {
      var v = el.type === 'checkbox' ? (el.checked ? '1' : '') : el.value.trim();
      if (el.getAttribute('data-filter') === 'sort' && v === 'new') v = '';
      if (v) params.set(key(el), v); else params.delete(key(el));
    });
    if (status !== 'all') params.set('status', status); else params.delete('status');
    if (pager && page > 1) params.set('page', page); else params.delete('page');
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
      } else if (name === 'unit' || name === 'day' || name === 'reviewed') {
        if (el.value && row.getAttribute('data-f-' + name) !== el.value) return false;
      } else if (name === 'site') {
        if (el.value && el.value.split('|').indexOf(row.getAttribute('data-f-site')) < 0) return false;
      } else if (name === 'by') {
        // Reviewed by someone, or ("!name") not by them (memory/74).
        if (!el.value) continue;
        var not = el.value.charAt(0) === '!';
        var who = not ? el.value.slice(1) : el.value;
        var by = (row.getAttribute('data-f-by') || '').split(' ').indexOf(who) >= 0;
        if (by === not) return false;
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
    ordered = sorted;
    if (parent) sorted.forEach(function (r) { parent.insertBefore(r, end); });
  }

  // The page of clips on screen, fetched as rows. A later page wins: an answer
  // that comes back after the filter moved on is dropped.
  var showing = null, asked = 0;
  function showPage(visible) {
    var size = Number(pager.getAttribute('data-page-size'));
    var pages = Math.max(1, Math.ceil(visible.length / size));
    page = Math.min(page, pages);
    pager.querySelector('[data-page-now]').textContent = page;
    pager.querySelector('[data-page-all]').textContent = pages;
    pager.querySelector('[data-page-prev]').disabled = page <= 1;
    pager.querySelector('[data-page-next]').disabled = page >= pages;
    var box = document.getElementById('clip-rows');
    var ids = visible.slice((page - 1) * size, page * size)
      .map(function (r) { return r.getAttribute('data-f-run'); });
    if (showing === null) {
      showing = Array.prototype.map.call(box.querySelectorAll('.clip-row'), function (r) {
        return r.getAttribute('data-f-run');
      }).join();
    }
    if (ids.join() === showing) return;
    showing = ids.join();
    var mine = ++asked;
    if (!ids.length) { box.innerHTML = ''; return; }
    var body = new URLSearchParams();
    ids.forEach(function (id) { body.append('run', id); });
    box.setAttribute('aria-busy', 'true');
    fetch(pager.getAttribute('data-url'), {method: 'POST', credentials: 'same-origin', body: body,
      headers: {'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value}})
      .then(function (r) { return r.ok ? r.text() : Promise.reject(r.status); })
      .then(function (html) {
        if (mine !== asked) return;
        box.innerHTML = html;
        box.removeAttribute('aria-busy');
        document.dispatchEvent(new CustomEvent('cliprows:change'));
      })
      .catch(function () {
        if (mine !== asked) return;
        showing = '';
        box.removeAttribute('aria-busy');
        box.innerHTML = '<p class="px-4 py-8 text-center text-[13px] text-red-700">Could not load these clips. Reload the page to try again.</p>';
      });
  }

  // The "Run these clips through a pipeline" form follows the filter, so
  // "re-run the clips with no tracks" is: filter, then Run.
  function updateRerun(visible, filtered) {
    var ids = Array.from(new Set(visible.map(function (r) { return r.getAttribute('data-f-video'); })
      .filter(Boolean)));
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

  // "Re-run these N": the form is further up the page, and a bare #rerun jump
  // went unnoticed (or nowhere, clicked twice). Bring it into view, mark it
  // for a moment and put the cursor on its pipeline menu.
  document.querySelectorAll('[data-rerun-jump]').forEach(function (a) {
    a.addEventListener('click', function (ev) {
      var form = document.getElementById('rerun');
      if (!form) return;
      ev.preventDefault();
      form.scrollIntoView({behavior: 'smooth', block: 'center'});
      var card = form.parentNode;
      card.classList.add('ring-2', 'ring-amber-500');
      setTimeout(function () { card.classList.remove('ring-2', 'ring-amber-500'); }, 2500);
      var pick = form.querySelector('select, [type=submit]');
      if (pick) pick.focus({preventScroll: true});
    });
  });

  // Sites: any number, in the filter bar and above the metrics, held in the one
  // hidden control as "a|b".
  var siteEl = bar.querySelector('[data-filter="site"]');
  function sitePicks() { return Array.prototype.slice.call(document.querySelectorAll('[data-site-pick]')); }
  function paintSites() {
    if (!siteEl) return;
    var on = siteEl.value ? siteEl.value.split('|') : [];
    sitePicks().forEach(function (b) { b.checked = on.indexOf(b.value) >= 0; });
  }
  document.addEventListener('change', function (ev) {
    if (!siteEl || !ev.target.matches('[data-site-pick]')) return;
    var on = siteEl.value ? siteEl.value.split('|') : [];
    on = on.filter(function (s) { return s !== ev.target.value; });
    if (ev.target.checked) on.push(ev.target.value);
    siteEl.value = on.join('|');
    fresh();
  });

  function apply() {
    paintSites();
    var sortEl = bar.querySelector('[data-filter="sort"]');
    sortRows(sortEl ? sortEl.value : 'new');
    var visible = [];
    ordered.forEach(function (r) {
      var ok = matches(r);
      r.style.display = ok ? '' : 'none';
      if (ok) visible.push(r);
    });
    if (pager) showPage(visible);
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
    // Others follow the filter (memory/73: the review metrics, the selection).
    document.dispatchEvent(new CustomEvent('clipfilter:change', {
      detail: {visible: visible, filtered: filtered, rows: rows}}));
  }

  function clear() {
    controls.forEach(function (el) {
      if (el.getAttribute('data-filter') === 'sort') return;
      if (el.type === 'checkbox') el.checked = false;
      else el.value = '';
    });
    status = 'all';
    fresh();
  }

  // A changed filter starts from its first page.
  function fresh() { page = 1; apply(); }

  controls.forEach(function (el) {
    el.addEventListener(el.tagName === 'SELECT' || el.type === 'checkbox' ? 'change' : 'input', fresh);
  });
  chips.forEach(function (c) {
    c.addEventListener('click', function () { status = c.getAttribute('data-filter-status'); fresh(); });
  });
  clearEls.forEach(function (el) { el.addEventListener('click', clear); });
  if (pager) {
    pager.querySelector('[data-page-prev]').addEventListener('click', function () { page -= 1; apply(); });
    pager.querySelector('[data-page-next]').addEventListener('click', function () { page += 1; apply(); });
  }

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
