// The insect grouping editor (memory/66): groups and the taxa in them on the
// left, every name the model gave in your runs on the right, a preview on a
// batch at the bottom. The grouping is plain data —
// {groups: [{name, color, members: [{rank, name}]}], tracks: [...]} — matched
// most specific first, the same rule as apps/pipelines/taxon_groups.py.
(function () {
  var root = document.getElementById('grouping');
  if (!root) return;
  var page = JSON.parse(root.getAttribute('data-page'));
  var RO = page.readonly;
  var RANKS = ['species', 'genus', 'family', 'order', 'class'];
  var UNCLEAR = 'Unclear';
  var state = clone(page.grouping);
  var tab = 'all', find = '', dirty = false;
  var $ = function (id) { return document.getElementById(id); };

  function clone(x) { return JSON.parse(JSON.stringify(x)); }
  function key(s) { return String(s || '').replace(/_/g, ' ').trim().replace(/\s+/g, ' ').toLowerCase(); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function csrf() { var i = root.querySelector('[name=csrfmiddlewaretoken]'); return i ? i.value : ''; }
  function status(text) { $('ge-status').textContent = text || ''; }
  function touch() { dirty = true; status('Not saved yet.'); render(); }

  // ── Matching ───────────────────────────────────────────────────────────
  function index(grouping) {
    var idx = {};
    grouping.groups.forEach(function (g) {
      g.members.forEach(function (m) { idx[m.rank + '|' + key(m.name)] = g.name; });
    });
    var bees = grouping.groups.filter(function (g) {
      return g.members.some(function (m) { return m.rank === 'family' && m.name === 'Apidae'; });
    })[0];
    idx.__bees__ = bees ? bees.name : null;
    return idx;
  }
  function match(row, idx) {
    var pairs = [['species', row.name], ['genus', row.genus], ['family', row.family],
                 ['order', row.order], ['class', row['class']]];
    for (var i = 0; i < pairs.length; i++) {
      if (pairs[i][0] === 'order' && row.bee && idx.__bees__) return [idx.__bees__, 'bee genus ' + row.genus];
      var v = pairs[i][1];
      if (v && idx[pairs[i][0] + '|' + key(v)]) return [idx[pairs[i][0] + '|' + key(v)], pairs[i][0] + ' ' + v];
    }
    return [UNCLEAR, 'not in a group'];
  }
  var defaultIdx = index(page['default']);

  function putMember(rank, name, group) {
    state.groups.forEach(function (g) {
      g.members = g.members.filter(function (m) { return !(m.rank === rank && key(m.name) === key(name)); });
    });
    if (group && group !== UNCLEAR) {
      var g = state.groups.filter(function (x) { return x.name === group; })[0];
      if (g) g.members.push({rank: rank, name: name});
    }
  }

  // ── Groups ─────────────────────────────────────────────────────────────
  function renderGroups(idx) {
    var box = $('ge-groups');
    box.innerHTML = '';
    var tally = {};
    page.names.forEach(function (r) {
      var g = match(r, idx)[0];
      tally[g] = tally[g] || {names: 0, tracks: 0};
      tally[g].names++; tally[g].tracks += r.tracks;
    });
    state.groups.forEach(function (g, gi) {
      var card = el('div', 'bg-white border border-gray-200 rounded-lg p-3 flex flex-col gap-2');
      var head = el('div', 'flex flex-wrap items-center gap-2');
      var color = el('input');
      color.type = 'color'; color.value = g.color || '#6b7280'; color.disabled = RO;
      color.className = 'w-8 h-8 rounded border border-gray-300 p-0.5 bg-white';
      color.setAttribute('aria-label', 'Colour of ' + g.name);
      color.addEventListener('change', function () { g.color = color.value; touch(); });
      var name = el('input', 'h-9 flex-1 min-w-[8rem] rounded border border-transparent hover:border-gray-300 focus:border-gray-400 px-2 font-semibold');
      name.value = g.name; name.maxLength = 60; name.disabled = RO;
      name.setAttribute('aria-label', 'Group name');
      name.addEventListener('change', function () {
        var v = name.value.trim().replace(/\s+/g, ' ');
        if (!v || v === UNCLEAR || state.groups.some(function (x) { return x !== g && x.name === v; })) {
          name.value = g.name; status('Pick a name no other group has.'); return;
        }
        (state.tracks || []).forEach(function (t) { if (t.group === g.name) t.group = v; });
        g.name = v; touch();
      });
      var t = tally[g.name] || {names: 0, tracks: 0};
      head.appendChild(color); head.appendChild(name);
      head.appendChild(el('span', 'text-xs text-gray-600', t.tracks + ' tracks · ' + t.names + ' names'));
      if (!RO) {
        var del = el('button', 'h-9 px-2 rounded border border-gray-300 bg-white hover:bg-gray-50 text-xs text-gray-700', 'Delete');
        del.type = 'button';
        del.addEventListener('click', function () {
          if (!window.confirm('Delete "' + g.name + '"? Its taxa will be in no group (Unclear) unless another group holds them.')) return;
          state.groups.splice(gi, 1);
          state.tracks = (state.tracks || []).filter(function (x) { return x.group !== g.name; });
          touch();
        });
        head.appendChild(del);
      }
      card.appendChild(head);

      var chips = el('div', 'flex flex-wrap gap-1.5');
      g.members.slice().sort(function (a, b) {
        return RANKS.indexOf(b.rank) - RANKS.indexOf(a.rank) || a.name.localeCompare(b.name);
      }).forEach(function (m) {
        var chip = el('span', 'inline-flex items-center gap-1.5 rounded-full bg-gray-100 pl-2.5 pr-1 py-0.5 text-xs');
        chip.appendChild(el('span', 'uppercase tracking-wide text-[10px] text-gray-500', m.rank));
        chip.appendChild(el('span', 'italic text-gray-900', m.name));
        if (!RO) {
          var x = el('button', 'w-6 h-6 rounded-full hover:bg-gray-200 text-gray-600', '×');
          x.type = 'button'; x.setAttribute('aria-label', 'Take ' + m.name + ' out of ' + g.name);
          x.addEventListener('click', function () { putMember(m.rank, m.name, ''); touch(); });
          chip.appendChild(x);
        }
        chips.appendChild(chip);
      });
      if (!g.members.length) chips.appendChild(el('span', 'text-xs text-gray-500', 'No taxa yet. Add one below, or pick this group for names on the right.'));
      card.appendChild(chips);

      if (!RO) {
        var add = el('form', 'flex flex-wrap gap-1.5');
        var rank = el('select', 'h-9 rounded border border-gray-300 bg-white px-1 text-xs');
        rank.setAttribute('aria-label', 'Rank');
        RANKS.forEach(function (r) { var o = el('option', null, r); o.value = r; if (r === 'family') o.selected = true; rank.appendChild(o); });
        var taxon = el('input', 'h-9 flex-1 min-w-[8rem] rounded border border-gray-300 px-2 text-xs');
        taxon.placeholder = 'e.g. Megachilidae'; taxon.setAttribute('aria-label', 'Taxon name');
        var go = el('button', 'h-9 px-2 rounded border border-gray-300 bg-white hover:bg-gray-50 text-xs', '+ taxon');
        go.type = 'submit';
        add.appendChild(rank); add.appendChild(taxon); add.appendChild(go);
        add.addEventListener('submit', function (e) {
          e.preventDefault();
          var v = taxon.value.trim().replace(/_/g, ' ').replace(/\s+/g, ' ');
          if (!v) return;
          putMember(rank.value, v, g.name); touch();
        });
        card.appendChild(add);
      }
      box.appendChild(card);
    });
  }

  // ── Names ──────────────────────────────────────────────────────────────
  function renderNames(idx) {
    var rows = page.names.map(function (r) {
      var m = match(r, idx), d = match(r, defaultIdx);
      return {r: r, group: m[0], why: m[1], mine: m[0] !== d[0]};
    });
    var none = rows.filter(function (x) { return x.group === UNCLEAR; }).length;
    var mine = rows.filter(function (x) { return x.mine; }).length;
    var tabs = $('ge-tabs');
    tabs.innerHTML = '';
    [['all', 'All names (' + rows.length + ')'], ['none', 'Not in a group (' + none + ')'],
     ['mine', 'Not where the default puts it (' + mine + ')']].forEach(function (t) {
      var b = el('button', 'h-10 px-3 rounded text-sm ' + (tab === t[0]
        ? 'border border-green-700 bg-green-50 text-green-800 font-semibold'
        : 'border border-gray-300 bg-white text-gray-700 hover:bg-gray-50'), t[1]);
      b.type = 'button'; b.setAttribute('aria-pressed', String(tab === t[0]));
      b.addEventListener('click', function () { tab = t[0]; render(); });
      tabs.appendChild(b);
    });
    var q = key(find);
    var shown = rows.filter(function (x) {
      if (tab === 'none' && x.group !== UNCLEAR) return false;
      if (tab === 'mine' && !x.mine) return false;
      if (!q) return true;
      return [x.r.name, x.r.genus, x.r.family, x.r.order].some(function (v) { return key(v).indexOf(q) !== -1; });
    });
    $('ge-names-count').textContent = page.names.length + ' names · '
      + page.names.reduce(function (n, r) { return n + r.tracks; }, 0) + ' tracks';
    var body = $('ge-names');
    body.innerHTML = '';
    var LIMIT = 300;
    shown.slice(0, LIMIT).forEach(function (x) {
      var tr = el('tr');
      tr.appendChild(el('td', 'px-4 py-1.5 italic', x.r.name));
      tr.appendChild(el('td', 'px-4 py-1.5 text-gray-600 text-xs', [x.r.family, x.r.order].filter(Boolean).join(' · ')
        || (x.r.bee ? 'a bee genus' : (x.r.pending ? 'looking up…' : 'not found'))));
      var td = el('td', 'px-4 py-1.5');
      var sel = el('select', 'h-9 w-44 rounded px-1 text-sm ' + (x.group === UNCLEAR ? 'border border-amber-500 bg-amber-50' : 'border border-gray-300 bg-white'));
      sel.disabled = RO; sel.setAttribute('aria-label', 'Group for ' + x.r.name);
      state.groups.map(function (g) { return g.name; }).concat([UNCLEAR]).forEach(function (name) {
        var o = el('option', null, name); o.value = name; o.selected = name === x.group; sel.appendChild(o);
      });
      sel.addEventListener('change', function () {
        // A species rule: this name only. Choosing Unclear takes it out of its group.
        if (sel.value === UNCLEAR) {
          putMember('species', x.r.name, '');
          if (match(x.r, index(state))[0] !== UNCLEAR) {
            status(x.r.name + ' is still in a group through its ' + match(x.r, index(state))[1].split(' ')[0]
              + '. Take that taxon out on the left to make it unclear.');
          }
        } else {
          putMember('species', x.r.name, sel.value);
        }
        touch();
      });
      td.appendChild(sel);
      tr.appendChild(td);
      tr.appendChild(el('td', 'px-4 py-1.5 text-right tabular-nums', String(x.r.tracks)));
      tr.appendChild(el('td', 'px-4 py-1.5 text-xs text-gray-600', x.why));
      body.appendChild(tr);
    });
    if (!page.names.length && !loading) {
      var empty = el('tr');
      var c = el('td', 'px-4 py-6 text-sm text-gray-600', 'No names yet: run a pipeline with Identify Species and they show here.');
      c.colSpan = 5; empty.appendChild(c); body.appendChild(empty);
    }
    $('ge-more').textContent = loading ? 'Loading the names from your runs…'
      : (pending ? 'Looking up the family of ' + pending + ' more name' + (pending === 1 ? '' : 's') + '… ' : '')
        + (shown.length > LIMIT ? 'Showing ' + LIMIT + ' of ' + shown.length + '. Use Find to narrow it.' : '');

    // Bulk: families whose names are in no group.
    var bulk = $('ge-bulk');
    bulk.innerHTML = '';
    if (RO) return;
    var fam = {};
    rows.forEach(function (x) { if (x.group === UNCLEAR && x.r.family) fam[x.r.family] = (fam[x.r.family] || 0) + 1; });
    Object.keys(fam).sort(function (a, b) { return fam[b] - fam[a]; }).slice(0, 3).forEach(function (f) {
      var line = el('form', 'flex flex-wrap items-center gap-2 text-sm bg-amber-50 border border-amber-200 rounded px-3 py-2');
      line.appendChild(el('span', null, fam[f] === 1 ? '1 name in ' + f + ' isn’t in a group.'
        : fam[f] + ' names in ' + f + ' aren’t in a group.'));
      var pick = el('select', 'h-9 rounded border border-gray-300 bg-white px-1');
      pick.setAttribute('aria-label', 'Group for family ' + f);
      state.groups.forEach(function (g) { var o = el('option', null, g.name); o.value = g.name; pick.appendChild(o); });
      var b = el('button', 'h-9 px-3 rounded border border-amber-600 bg-white hover:bg-amber-100', 'Put family ' + f + ' in it');
      b.type = 'submit';
      line.appendChild(pick); line.appendChild(b);
      line.addEventListener('submit', function (e) { e.preventDefault(); putMember('family', f, pick.value); touch(); });
      bulk.appendChild(line);
    });
  }

  function render() {
    var idx = index(state);
    renderGroups(idx);
    renderNames(idx);
  }

  // ── Save, copy, reset ──────────────────────────────────────────────────
  function save() {
    return fetch(page.save_url, {method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf()},
      body: JSON.stringify({grouping: state})})
      .then(function (r) { return r.json(); })
      .then(function (res) {
        if (!res.ok) throw new Error(res.error || 'Could not save.');
        dirty = false;
        status('Saved to the pipeline. Batches that already ran keep their counts until you apply it to them.');
        return res;
      });
  }
  if (!RO) {
    $('ge-save').addEventListener('click', function () { save().catch(function (e) { status(e.message); }); });
    $('ge-new-group').addEventListener('submit', function (e) {
      e.preventDefault();
      var v = $('ge-new-name').value.trim().replace(/\s+/g, ' ');
      if (!v || v === UNCLEAR || state.groups.some(function (g) { return g.name === v; })) { status('Pick a name no other group has.'); return; }
      var palette = ['#7c3aed', '#0891b2', '#ca8a04', '#be185d', '#4d7c0f', '#9333ea', '#0f766e'];
      state.groups.push({name: v, color: palette[state.groups.length % palette.length], members: []});
      $('ge-new-name').value = '';
      touch();
    });
    var copy = $('ge-copy');
    page.others.forEach(function (o, i) { var opt = el('option', null, o.title); opt.value = String(i); copy.appendChild(opt); });
    if (!page.others.length) copy.hidden = true;
    copy.addEventListener('change', function () {
      var o = page.others[Number(copy.value)];
      copy.value = '';
      if (!o || !window.confirm('Replace this grouping with the one from "' + o.title + '"?')) return;
      state = clone(o.grouping); state.tracks = state.tracks || []; touch();
    });
    $('ge-reset').addEventListener('click', function () {
      if (!window.confirm('Go back to the default groups? Your groups and taxa here are replaced (single tracks you set while reviewing are kept).')) return;
      var tracks = state.tracks || [];
      state = clone(page['default']); state.tracks = tracks; touch();
    });
    window.addEventListener('beforeunload', function (e) { if (dirty) { e.preventDefault(); e.returnValue = ''; } });

    // ── Try it on a batch ──────────────────────────────────────────────
    var batch = $('ge-batch');
    page.batches.forEach(function (b) {
      var o = el('option', null, b.name + ' (' + b.clips + ' clips)'); o.value = b.id; batch.appendChild(o);
    });
    if (!page.batches.length) {
      batch.appendChild(el('option', null, 'No batches of this pipeline yet'));
      batch.disabled = true; $('ge-try').disabled = true;
    }
    $('ge-try').addEventListener('click', function () {
      var box = $('ge-preview');
      box.innerHTML = '';
      box.appendChild(el('p', 'text-sm text-gray-600', 'Working it out from the saved crop readings…'));
      $('ge-apply').hidden = true;
      fetch(page.preview_url, {method: 'POST', credentials: 'same-origin',
        headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf()},
        body: JSON.stringify({grouping: state, batch: batch.value})})
        .then(function (r) { return r.json(); })
        .then(function (res) {
          box.innerHTML = '';
          if (!res.ok) { box.appendChild(el('p', 'text-sm text-red-700', res.error || 'Could not preview.')); return; }
          var t = el('table', 'min-w-full text-sm');
          var head = el('tr', 'bg-gray-50 text-xs uppercase tracking-wide text-gray-600');
          ['Group', 'Tracks now', 'With this grouping', 'Counted'].forEach(function (h) { head.appendChild(el('th', 'px-3 py-2 text-left font-semibold', h)); });
          t.appendChild(head);
          res.rows.forEach(function (r) {
            var tr = el('tr', 'border-t border-gray-100');
            tr.appendChild(el('td', 'px-3 py-1.5 font-medium', r.group));
            tr.appendChild(el('td', 'px-3 py-1.5 text-gray-600', String(r.now)));
            var d = r.after - r.now;
            tr.appendChild(el('td', 'px-3 py-1.5', r.after + (d ? ' (' + (d > 0 ? '+' : '') + d + ')' : '')));
            tr.appendChild(el('td', 'px-3 py-1.5 ' + (r.counted ? 'text-green-800' : 'text-orange-800'), r.counted ? 'counted' : 'left out'));
            t.appendChild(tr);
          });
          box.appendChild(t);
          box.appendChild(el('p', 'mt-2 text-sm text-gray-700', 'Events counted: ' + res.events_now + ' → ' + res.events_after
            + ' · from ' + res.clips + ' of ' + res.of + ' clips'
            + (res.skipped ? ' (' + res.skipped + ' had no crop readings to use, or ran past the time limit)' : '') + '.'));
          var form = $('ge-apply');
          form.action = page.apply_url.replace('00000000-0000-0000-0000-000000000000', batch.value);
          $('ge-apply-note').textContent = 'Recounts the batch in place. No GPU-hours.';
          form.hidden = false;
        })
        .catch(function () { box.innerHTML = ''; box.appendChild(el('p', 'text-sm text-red-700', 'Could not preview.')); });
    });
    $('ge-apply').addEventListener('submit', function (e) {
      e.preventDefault();
      var form = e.currentTarget;
      save().then(function () { dirty = false; form.submit(); }).catch(function (err) { status(err.message); });
    });
  }
  $('ge-find').addEventListener('input', function (e) { find = e.target.value; render(); });

  state.tracks = state.tracks || [];

  // The names list comes after the page (it reads all your runs). Names not
  // yet looked up in GBIF arrive over a few more fetches.
  var loading = true, pending = 0, tries = 0;
  function loadNames() {
    fetch(page.names_url, {credentials: 'same-origin'})
      .then(function (r) { if (!r.ok) throw new Error(); return r.json(); })
      .then(function (res) {
        page.names = res.names || [];
        pending = res.pending || 0;
        loading = false;
        render();
        if (pending && ++tries < 15) setTimeout(loadNames, 800);
      })
      .catch(function () {
        loading = false;
        $('ge-more').textContent = 'Could not load the names. Reload the page to try again.';
      });
  }
  page.names = page.names || [];
  render();
  loadNames();
})();
