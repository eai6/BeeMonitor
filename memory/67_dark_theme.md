# 67 — A dark theme for the platform

**Status:** built (2026-10-31) with the proposed answers. Design canvas: https://claude.ai/artifact/JyNDxteD9GW6suXjuon4GX (boards "Batch page: light and dark", "Theme menu in the navbar"). Requested by the user.

> Write under any **Feedback:** line. *Proposed* means a starting point, not a decision.

## What exists (audit, 2026-10-31)
- **Tailwind:**
  - It comes from the Play CDN (`cdn.tailwindcss.com`), with no build step.
  - Two bases, `templates/base.html` and `templates/public/base.html`, set `tailwind.config` inline (`amber` is remapped to the green brand palette) with no `darkMode`. Both hard-code `<html class="h-full bg-gray-50">` and `<meta name="theme-color" content="#16a34a">`.
  - `platform_admin` has its own layout (`apps/platform_admin/templates/platform_admin/_layout.html`).
- **No dark support anywhere:** no `dark:` classes and no `prefers-color-scheme`.
- **Hard-coded light utilities:** 97 of 100 templates, about 3,100 occurrences of `bg-white`, `bg-gray-*`, `text-gray-*` and `border-gray-*`. The biggest are `devices/detail.html` (168), `pipelines/batch.html` (82), `accounts/usage.html` (74), `_track_overlay.html` (71), `training/detail.html`, `analysis/results.html`, `videos/detail.html` and the annotation tabs.
- **Inline colours:** 25 `style=""` colours.
- **`<style>` blocks with hex colours:** `pipelines/editor.html` (47 hexes, Drawflow nodes), `annotations/editor.html`, `support_chat.html`, `agent_chat.html`, `setup/walkthrough.html`, `navbar.html`, `_share_panel.html` and `_cancel_batch.html`.
- **JS colours:**
  - `period_review.js`: about 10 hexes (`#fff` rings, a `#f3f4f6`/`#e5e7eb` stripe);
  - `clip_filter.js`: `bg-gray-100 text-gray-700`;
  - `clip_viewer.js`: `text-gray-500`, `bg-amber-50`;
  - `track_overlay.js` draws on video and is theme-neutral;
  - `add_missed.js` is already dark.
- **Third-party styling:**
  - Drawflow's CSS, plus our overrides;
  - Chart.js on `devices/detail.html` (4 charts, default grid and text colours);
  - Leaflet with OSM tiles (`devices/list.html`, `videos/upload.html`);
  - highlight.js `github.min.css` (light), in the chats and the walkthrough.
- **Settings:** `UserProfile` has no preferences field. `localStorage` already holds UI preferences (`bm-ov-bytube`, `cv-regions`, `pr-regions`).
- **Emails:** `accounts/emails.py`, which renders HTML templates.

## The plan

### 1. The switch
- **Theme menu** in the navbar's user menu: **System** (the default, follows the device), **Light**, **Dark**. It's kept in `localStorage` (`bm-theme`). *Proposed:* also a `UserProfile.theme` field so it follows the user across devices. That needs a migration, so it can come later.
- **No flash:** a tiny inline script in `<head>`, before Tailwind loads, sets `html.dark` from the stored choice or `prefers-color-scheme`, and listens for the system changing.
- **Config:** `darkMode: 'class'` in both bases. `theme-color` switches to a dark green on dark.

### 2. Colours: one override layer, not 3,100 edits
- **The layer:** a `static/css/dark.css` (the first file in `static/css`) loaded by both bases. It remaps the light utilities under `html.dark`:
  - `bg-white` and the card surfaces become `#111827`;
  - `bg-gray-50`, the page, becomes `#0b0f14`;
  - `bg-gray-100`/`200` become raised surfaces;
  - `text-gray-900`…`400` move to a light scale that keeps 4.5:1;
  - `border-gray-*` become dark borders;
  - the tinted panels (`bg-green-50`, `bg-amber-50`, `bg-red-50`, `bg-blue-50`, `bg-orange-50`) become dark tints with light text.
  - It also covers `hover:` variants, form controls (inputs, selects, textareas, checkboxes with `color-scheme: dark`), tables, `shadow`s and `<dialog>`.
- **Why a layer:** dark mode arrives on all 97 templates at once, and new pages get it for free. A template that needs something special can still use `dark:` classes.
- **Brand:** green buttons stay green. Links lighten one step so they pass 4.5:1 on dark.

### 3. Things the layer can't reach
- **Pipeline canvas (Drawflow):** node, port, label and the type-chip colours move to CSS variables with dark values.
- **`<style>` blocks with hexes:** navbar, support/agent chat, walkthrough, share and cancel panels, annotation editor. Each moves to variables.
- **JS:**
  - `period_review.js`: rings and stripes read CSS variables;
  - `clip_filter.js` and `clip_viewer.js`: their injected classes are covered by the layer, checked.
- **Chart.js** (`devices/detail.html`): grid and tick colours from the theme, redrawn when it changes.
- **highlight.js:** swap `github.min.css` for `github-dark.min.css` on dark.
- **Leaflet:** *proposed:* tiles stay as they are (maps read better light); only the controls and popups go dark.
- **The video player:** video, boxes and the overlay stay as they are. Only the frame around them changes.

### 4. What stays light
- **Emails** (many clients ignore dark CSS).
- **The rendered annotated video.**
- *Proposed:* the public share pages **follow the viewer's system setting**, since they have no menu.

## Build order
1. **The switch:** head script, `darkMode: 'class'`, the theme menu, `theme-color`.
2. **`dark.css`** for the common utilities, tints, forms, tables and dialogs. Check the 10 biggest pages.
3. **Drawflow, the `<style>` blocks, `period_review.js`, Chart.js, highlight.js, Leaflet controls.**
4. **A sweep of every page** for anything unreadable (contrast 4.5:1), the public pages, and `platform_admin`.

## Questions
1. **Default:** follow the device's setting (proposed), or always start light?
2. **Remember the choice:** in the browser only (proposed, for now), or on the account so it follows you to other devices (a migration)?
3. **Public share pages:** follow the viewer's system setting (proposed), or always light?

**Feedback:**

## Built (2026-10-31)
- **`templates/components/theme_head.html`**, included in both bases right after `theme-color`:
  - sets `html.dark` before paint;
  - System follows `prefers-color-scheme` and its changes; Light or Dark is kept in `localStorage` `bm-theme`;
  - swaps highlight.js `github` ↔ `github-dark`;
  - sets `theme-color`;
  - exposes `window.bmTheme` (`get`, `set`, `isDark`) and a `bm-theme` event;
  - `darkMode: 'class'` in both Tailwind configs.
- **`static/css/dark.css`:** the override layer for surfaces, text, borders and dividers, tints and coloured text, forms, tables, dialogs, chat markdown, the navbar green and Leaflet controls. It also sets the `--bm-*` variables.
- **Navbar:** a theme button (System → Light → Dark), on desktop and in the mobile menu.
- **Specific fixes:**
  - Drawflow nodes and type chips (`editor.html`);
  - `period_review.js` rings and the stripe, plus the stripe in `period_review.html`, through `--bm-*`;
  - Chart.js defaults on `devices/detail.html`.
- **Unchanged:** emails, canvas drawings over video and images, and map tiles. Public share pages follow the visitor's device.
- **Not yet done:** a page-by-page contrast sweep in a real browser (step 4 of the build order).

