# 59 — Publishing pipelines, like datasets and models

**Status:** built (2026-10-27), not committed. The open questions went as planned: others can run it directly, and a private model blocks publishing. Also: delete is refused while other people have runs on the pipeline (runs CASCADE); device schedules stay limited to your own pipelines and templates.
Canvas: https://claude.ai/artifact/JD3BmQMawK1KdY8cTKXhdi
Related: [[37_project_sharing_and_publishing]] (datasets), [[47_public_batch_share_link]] (results).

## Why
The user wants to share pipelines the way models and annotation datasets are shared, so other people can review them or use them in their research.

## Audit (2026-10-27)
- `Pipeline` has `user`, `title`, `description`, `steps` (JSON), `graph` (Drawflow layout), `is_template`, `cloned_from`. It has no visibility.
- **Templates are already a public pipeline in all but name.** Anyone can run one (`views.py` run check: `is_template or owner`) and copy one (`clone_pipeline`). Only staff can make them.
- **Running snapshots the steps.** `PipelineRun.steps` is a copy, so if a published pipeline is edited later, runs other people already made still say what they ran.
- **Steps can point at private models.** Detect's `object_model` / `reference_model` hold a `CustomModel` pk. The executor resolves the pk through `CustomModel.usable(run.user)` (own models or public ones). A private model therefore fails for everyone else with "The selected … model is unavailable".
- **Nothing else in a step is personal.** Clips, devices, sites and ROIs are picked at run time on the Processing page, not stored in the pipeline.
- **Datasets** are `visibility` + `published_at` + `publish_metadata`, listed on Browse › Datasets, and copied, never edited by strangers (memory/37). **Models** are `visibility` + `published_at`, listed on Browse › Models, and used directly.
- **Batch share links** (memory/47) share a run's *results*. Publishing a pipeline shares the *method*. Neither one exposes the other.

## Plan
1. `Pipeline.visibility` (private/public), `published_at`, and `Pipeline.public()`. Migration only; templates are unchanged.
2. **Publish / unpublish** from the editor header (owner only). Publishing is refused while a step uses a private model; the message names the model and links to publish it, so a published pipeline always runs for the people it is shared with.
3. **Browse › Pipelines**, a third tab. Each card shows:
   - title, author, description, last updated;
   - the steps in order (block icons and names) and the models they use;
   - copies taken.

   Buttons: **View** (read-only) and **Copy to my pipelines**.
4. **Read-only view.** The editor opens a pipeline you don't own in view mode: no palette, no save, the Drawflow canvas locked, and a banner "Published by X · Copy to change it".
5. **Use.** Any public pipeline can be:
   - run directly, the way a template can (the run check becomes `owner or template or public`);
   - picked on the Processing page under a "Published" group;
   - copied, which records `cloned_from`. A copy says "Copied from X by Y".
6. **Unpublish.** Stops listing and new runs. Copies and past runs are kept.
7. **Not now:** sharing with named people (no co-editing); version history and citing a fixed version (the run's step snapshot covers reproducibility); comments.

## Tests
- Publishing refused while a private model is used; allowed once the model is public.
- A public pipeline is viewable, copyable and runnable by others, but not editable or saveable. A private one is none of these.
- Unpublish removes it from Browse and stops new runs, and keeps copies.
- Browse lists only public pipelines; the read-only editor has no save.

## Open questions
1. Should others be able to **run** a published pipeline directly, or only **copy** it first? Planned: run directly, like a template (datasets copy, models are used).
2. If a pipeline uses a private model, should we **refuse until the model is published** (planned), or offer "publish the model too" in the same step?
