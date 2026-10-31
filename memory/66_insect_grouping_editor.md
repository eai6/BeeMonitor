# 66 — An editor for insect groupings

**Status:** built (2026-10-31) with the proposed answers. Design canvas: https://claude.ai/artifact/NzvjTPKhCLDEzp9QXeTRvy (boards "Grouping editor", "Identify Species node", "Try on a batch"). Follows memory/65.

The request: a user should see and change which taxa make up each group, make their own groups, and organise the taxa their way. That doesn't fit inside a canvas node (the node in the screenshot was already crowded: checklist, share, rules, an Edit dialog). It needs its own page.

> Write under any **Feedback:** line. *Proposed* means a starting point, not a decision.

## What exists (memory/65, built 2026-10-31)
- **Built-in groups are code, not data.** `taxon_groups.default_group()` maps family/order/class to Bees, Wasps, Ants, Flies, Beetles, Moths & butterflies and Other crawlers. A user can't see it, rename a group, or take a family out of one.
- **Overrides are rules in the node.** `group_rules` = `{groups, rules}`, edited through a small dialog in the canvas editor. Rule kinds: name, genus, family, order, track.
- **Leave-out checklist** on the node; Unclear is always counted.
- **Names seen:** each run's Identify Species output keeps `names_seen` and `track_groups` (with `top` and `family`). That's enough to list every name a user's runs produced, with track counts, without a GPU.
- **Applying:** the batch page's **Apply the pipeline's groups to this batch** (`regroup.apply_pipeline`); changing a group while reviewing (`regroup.set_group`).
- **Bug:** a selected canvas node shows red. Drawflow's default `.selected` background shows through; our CSS only sets the border.

## The plan

### 1. A grouping is explicit and editable
- **Shape:** the Identify Species node's config gets `grouping` = `{"groups": [{"name", "color", "members": [{"rank", "name"}]}]}`. Ranks: class, order, family, genus, species. Track rules stay as they are.
- **Matching:** the most specific member wins (species, then genus, family, order, class). A name no member covers is Unclear.
- **Default:** a new node starts with the built-in groups written out as members:
  - Bees = families Apidae, Halictidae, Megachilidae, Andrenidae, Colletidae, Melittidae, Stenotritidae;
  - Ants = family Formicidae;
  - Wasps = order Hymenoptera;
  - and so on.

  So the default is visible and editable, not hidden in code. BeeMachine's bee genera stay a lookup fallback (when GBIF can't place a name), not a hidden group rule.
- **Old pipelines:** existing `group_rules` and the built-in defaults are converted once, on read, into `grouping`, so nothing counts differently until the user changes something.
- **"Wasps = order Hymenoptera":** this works because a more specific member wins: family Formicidae beats order Hymenoptera, so an ant stays an ant. The editor says why each name landed where it did.

### 2. The grouping editor page
- **Where:** `Pipelines › <pipeline> › Identify Species › Grouping` (`/pipelines/<pk>/grouping/<step_id>/`). The node shows a summary ("6 groups · 31 taxa · 4 set by you") and **Edit grouping ↗**. The grouping stays in the pipeline, as the user asked (memory/65 answer 4).
- **Left, groups:** a card per group with its colour, name, member taxa as chips (rank + name, removable), **+ taxon** (type a name; GBIF fills in its rank), and its tracks and names counts. **+ New group**; rename; delete (its members become unassigned).
- **Right, names the model gave in your runs:** one row per name from `names_seen` across the user's runs:
  - family · order;
  - **In group** (a select: picking a group adds a species member);
  - track count;
  - **Because** ("family Megachilidae", "species rule (you)", "not in a group").
  - Tabs: All / Not in a group / Set by you; search by species, genus or family.
- **Bulk:** "6 names in Formicidae aren't in a group yet → Put family Formicidae in Ants".
- **Copy from another pipeline:** start from another pipeline's grouping.
- **Save and try on a batch:** see §3.

### 3. Try it, then apply
- A **preview** (no save to runs) for a chosen batch: tracks per group now vs with this grouping, what's counted, and events counted before → after.
  - Computed from each run's saved crop readings, with a 60 s budget.
  - Big batches preview a sample of clips and say so.
- **Apply to this batch** reuses `regroup.apply_pipeline` (first 8 now, the rest by the sweep). No GPU.

### 4. Reviewing
- The player's **Change its group** stays. "Every crop read as X" adds a species member to the chosen group; "its genus/family" adds that member. A single track stays a track rule.
- The panel's group list follows the grouping's groups.

### 5. Small fixes in the same build
- The red selected node: white background, green ring.
- The node's **Edit** dialog is replaced by **Edit grouping ↗**.

## Build order
1. **`grouping` shape:** conversion from `group_rules` and defaults, and most-specific matching in `taxon_groups`. Tests: old configs count the same, species beats genus beats family beats order, Formicidae beats Hymenoptera.
2. **Editor page:** groups (add, rename, delete, members), names-seen table, filters, bulk by family, copy from a pipeline. Owner only; read-only for a published pipeline someone else owns.
3. **Preview and Apply.**
4. **Review panel and node:** wire both to the grouping; fix the red node.

## Questions
1. **Where groupings live:** keep each one in its pipeline (proposed, as you said), or make them shareable objects that several pipelines can use?
2. **Wasps:** should "Wasps" by default be "order Hymenoptera minus what's more specific" (proposed), or only the wasp families (Vespidae, Crabronidae, Sphecidae, Pompilidae…), leaving sawflies and others Unclear?
3. **Names-seen list:** should it show names from all your runs (proposed), or only the runs of this pipeline?

**Feedback:**

## Built (2026-10-31)
- **`taxon_groups`:**
  - `default_grouping()` writes the built-in groups out as members;
  - `grouping_of(config)` reads `grouping`, or converts the memory/65 `group_rules`;
  - `name_group` matches most specific first (species, genus, family, order, class). A bee genus GBIF couldn't place goes to the group holding Apidae, before the order check;
  - `with_rule` writes into the grouping. "Undo" puts a taxon back where the default had it;
  - `track_groups` keeps each track's crop `names` for previews.
- **Node:**
  - the `grouping` field (a summary plus **Edit grouping ↗**, which saves the canvas first) replaces the rules dialog;
  - the leave-out checklist lists the grouping's groups;
  - the canvas writes old rules out as a grouping on load (`views._with_groupings`).
- **Page** (`apps/pipelines/grouping_views.py`, `pipelines/grouping.html`, `static/js/grouping_editor.js`):
  - `pipelines:grouping`: GET for viewers; POST to save, owner only;
  - groups with colour, rename, delete, and add a taxon by rank;
  - names from the user's last 3000 runs, with All / Not in a group / Not where the default puts it, and Find;
  - per-family bulk suggestions; copy from another pipeline; start from the defaults.
- **Preview:** `pipelines:grouping_preview` gives tracks per group now vs after, and events before → after, with a 40 s budget. Then **Save and apply to this batch** (`batch_apply_groups`).
- **Fix:** a selected canvas node is white, not red.
- **Tests:** in `test_insect_groups.py` (39).
