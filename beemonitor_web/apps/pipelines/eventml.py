"""Event classifiers trained and evaluated on review data (memory/70).

Every reviewed detected event is a labelled example: accepted = a real event,
rejected = not. Its features come from its track on the run's current tracks
(the ones the reviewer saw) and its reference, exactly as the Events node
scores events (``ml_scoring``). Exits and entries train one model (the paper's
20 features); stays another (their own features).

The test set is always the **left-out days** (memory/57): never trained on,
and never split by event — clips share bees, light and layout. Validation on
the training days is cross-validated grouped by day. Missed events reviewers
added can't train a scorer of detected events, but count against recall.
"""

from __future__ import annotations

import io
import logging
from datetime import date

logger = logging.getLogger(__name__)

KINDS = {"exit_enter": ("exit", "enter"), "stay": ("stay",)}
CUT = 0.5


# ── Storing models ──────────────────────────────────────────────────────────

def store_model(payload: bytes, user_id, name) -> str:
    """Upload a trained model; its storage key."""
    import uuid

    from config.storage import get_s3_client

    key = f"event_models/{user_id}/{uuid.uuid4().hex}-{name}.joblib"
    get_s3_client().upload_stream("processed", key, io.BytesIO(payload),
                                  content_type="application/octet-stream")
    return key


def read_model(key) -> bytes:
    from config.storage import get_s3_client

    buf = io.BytesIO()
    get_s3_client().download_to_stream("processed", key, buf)
    return buf.getvalue()


# ── Left-out days ───────────────────────────────────────────────────────────

def default_left_out(user) -> str:
    """The left-out days of the user's annotation projects, one text."""
    from apps.annotations.models import AnnotationProject

    lines = []
    for text in AnnotationProject.objects.filter(user=user).values_list("left_out_days", flat=True):
        for line in (text or "").splitlines():
            if line.strip() and line.strip() not in lines:
                lines.append(line.strip())
    return "\n".join(lines)


# ── The training set ────────────────────────────────────────────────────────

def batches_with_reviews(user):
    """``[{id, name, reviewed}]``: the user's batches with reviewed events."""
    from django.db.models import Count

    from . import batches
    from .models import EventReview

    counts = dict(EventReview.objects.filter(run__user=user, run__batch_id__isnull=False)
                  .values_list("run__batch_id").annotate(n=Count("id"))
                  .values_list("run__batch_id", "n"))
    names = batches.names(list(counts))
    return [{"id": str(b), "name": names.get(b) or f"batch {str(b)[:8]}", "reviewed": n}
            for b, n in sorted(counts.items(), key=lambda kv: -kv[1])]


def dataset(user, batch_ids, left_out_text=""):
    """``(rows, summary)``: one row per reviewed detected event of the batches.

    A row: ``{kind, label, action, features, site, day, left_out, clip,
    batch, run, key}``. Disputed events are left out and counted; missed
    events are counted per action, day and site, for recall."""
    from apps.annotations import leftout

    from . import executors, ml_scoring, review
    from .models import AddedEvent, PipelineRun

    rules = leftout.parse(left_out_text)[0]
    from . import runsummary

    # Which clips were reviewed, from their slim copies; then each reviewed
    # one whole, one at a time (memory/76).
    runs = runsummary.slim_runs(PipelineRun.objects.filter(
        user=user, batch_id__in=list(batch_ids), status=PipelineRun.Status.COMPLETED))
    votes = review.review_votes(runs)
    added_by_run = {}
    for run_id, action in AddedEvent.objects.filter(run__in=runs).values_list("run_id", "action"):
        added_by_run.setdefault(run_id, []).append(action)
    rows, disputed, unscorable = [], 0, 0
    missed = []
    for slim in runs:
        mine = votes.get(slim.pk) or {}
        added = added_by_run.get(slim.pk, [])
        if not mine and not added:
            continue
        run = PipelineRun.objects.get(pk=slim.pk)
        steps = run.steps or []
        idx = next((i for i, s in enumerate(steps) if s.get("block_type") == "analyze.events"), None)
        if idx is None:
            continue
        step = steps[idx]
        inputs = executors.resolve_inputs(step, steps, idx, run.context or {})
        _tidy, refs, result, _fps, _src, _ref_src, extra = executors._analysis_inputs(
            step, run, run.context or {}, inputs, idx)
        video = extra.get("video")
        site = leftout.video_site(video) if video else ""
        day = leftout.video_day(video) if video else None
        held = bool(video and leftout.is_left_out(video, rules))
        for action in added:
            missed.append({"action": action, "site": site, "day": day, "left_out": held})
        events = {review.event_key(r): r for k, o in _analyzer_events(run) for r in o}
        paths, _why = ml_scoring.track_paths(result, video) if mine else (None, None)
        boxes = ml_scoring.ref_boxes(refs, video) if video else {}
        for key, verdicts in mine.items():
            row = events.get(key)
            if row is None:
                continue
            call = review.combined(verdicts)
            if call == "disputed":
                disputed += 1
                continue
            got = ml_scoring.features_for(row, paths or {}, boxes)
            if got is None:
                unscorable += 1
                continue
            kind, feats = got
            rows.append({"kind": kind, "label": int(call == "accepted"), "action": row["action"],
                         "features": feats, "site": site, "day": day, "left_out": held,
                         "clip": getattr(video, "pk", None), "batch": str(run.batch_id),
                         "run": str(run.pk), "key": key})
    return rows, _summarize(rows, disputed, unscorable, missed)


def _analyzer_events(run):
    from .aggregate import analyzer_outputs

    for kind, out in analyzer_outputs(run):
        if kind == "events":
            yield kind, out.get("rows") or []


def _summarize(rows, disputed, unscorable, missed):
    by_action = {}
    for r in rows:
        a = by_action.setdefault(r["action"], {"events": 0, "accepted": 0, "test": 0})
        a["events"] += 1
        a["accepted"] += r["label"]
        a["test"] += int(r["left_out"])
    for a in by_action.values():
        a["accepted_pct"] = round(100 * a["accepted"] / a["events"]) if a["events"] else None
    days = {}
    for r in rows:
        if r["day"] is not None:
            days.setdefault(r["site"] or "?", set()).add(r["day"])
    return {
        "events": len(rows), "by_action": by_action, "disputed": disputed,
        "unscorable": unscorable, "missed": len(missed), "missed_rows": missed,
        "train_days": len({(r["site"], r["day"]) for r in rows if not r["left_out"]}),
        "test_days": len({(r["site"], r["day"]) for r in rows if r["left_out"]}),
        "days_by_site": {s: len(d) for s, d in days.items()},
        "test_events": sum(1 for r in rows if r["left_out"]),
    }


def csv_text(rows):
    """The training set as CSV: features, label and where each came from."""
    import csv

    names = sorted({k for r in rows for k in r["features"]})
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["kind", "action", "label", "left_out", "site", "day", "clip", "batch", "run",
                "event_key"] + names)
    for r in rows:
        w.writerow([r["kind"], r["action"], r["label"], int(r["left_out"]), r["site"],
                    r["day"].isoformat() if isinstance(r["day"], date) else "", r["clip"],
                    r["batch"], r["run"], r["key"]] + [r["features"].get(n, "") for n in names])
    return buf.getvalue()


# ── Training and evaluation ─────────────────────────────────────────────────

def _features(kind):
    from . import ml_scoring

    if kind == "stay":
        return list(ml_scoring.STAY_FEATURES)
    return ["trajectory_length", "path_length", "displacement", "tortuosity", "avg_speed",
            "max_speed", "speed_std", "speed_cv", "start_speed", "middle_speed", "end_speed",
            "decel_ratio", "start_to_nest", "end_to_nest", "approach_ratio", "x_var", "y_var",
            "vertical_movement", "horizontal_movement", "is_entry"]


def _estimator(algorithm):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    if algorithm == "gradient_boosting":
        return GradientBoostingClassifier(random_state=0)
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  class_weight="balanced", random_state=0, n_jobs=1)


def evaluate(probs, labels, missed=0, cut=CUT):
    """Per-event and pipeline numbers for scores ``probs`` against ``labels``.

    Pipeline: if events below ``cut`` were dropped, precision among those kept
    and recall over every real event — accepted ones and missed ones added."""
    from sklearn.metrics import roc_auc_score

    n = len(labels)
    if not n:
        return None
    kept = [(p, y) for p, y in zip(probs, labels) if p >= cut]
    tp = sum(y for _p, y in kept)
    fp = len(kept) - tp
    real = sum(labels)
    fn = real - tp
    out = {
        "events": n, "real": real,
        "auc": (round(float(roc_auc_score(labels, probs)), 3)
                if 0 < real < n else None),
        "precision": round(100 * tp / (tp + fp)) if tp + fp else None,
        "recall": round(100 * tp / (tp + fn)) if tp + fn else None,
        "f1": (round(100 * 2 * tp / (2 * tp + fp + fn)) if tp else 0),
        "pipeline_precision": round(100 * tp / len(kept)) if kept else None,
        "pipeline_recall": round(100 * tp / (real + missed)) if real + missed else None,
        "kept": len(kept),
    }
    # Calibration: are 0.8s right 80% of the time?
    bins = []
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        b = [(p, y) for p, y in zip(probs, labels) if lo <= p < lo + 0.2 or (lo == 0.8 and p == 1)]
        if b:
            bins.append({"from": lo, "to": round(lo + 0.2, 1), "events": len(b),
                         "mean_score": round(sum(p for p, _ in b) / len(b), 2),
                         "real_share": round(sum(y for _, y in b) / len(b), 2)})
    out["calibration"] = bins
    return out


def train(rows, summary, kind, algorithm="random_forest"):
    """``(model, report)`` for one kind ("exit_enter" or "stay"); model None
    when there isn't enough to train on (the report says why)."""
    import pandas as pd

    names = _features(kind)
    data = [r for r in rows if r["kind"] == kind]
    train_rows = [r for r in data if not r["left_out"]]
    test_rows = [r for r in data if r["left_out"]]
    report = {"kind": kind, "algorithm": algorithm, "features": names,
              "train_events": len(train_rows), "test_events": len(test_rows),
              "train_days": len({(r["site"], r["day"]) for r in train_rows}),
              "test_days": len({(r["site"], r["day"]) for r in test_rows})}
    labels = [r["label"] for r in train_rows]
    if len(train_rows) < 10 or len(set(labels)) < 2:
        report["note"] = ("Not enough reviewed events to train on outside the left-out days: "
                          f"{len(train_rows)}, and both real and not-real ones are needed.")
        return None, report
    X = pd.DataFrame([r["features"] for r in train_rows])[names]
    model = _estimator(algorithm)

    # Validation on the training days, grouped by day.
    groups = [f"{r['site']}|{r['day']}" for r in train_rows]
    n_groups = len(set(groups))
    if n_groups >= 3:
        from sklearn.model_selection import GroupKFold, cross_val_predict

        try:
            cv = cross_val_predict(_estimator(algorithm), X, labels, groups=groups,
                                   cv=GroupKFold(n_splits=min(5, n_groups)),
                                   method="predict_proba")[:, 1]
            report["validation"] = evaluate(list(cv), labels)
        except ValueError as exc:      # a fold with one class
            report["validation_note"] = f"Grouped validation skipped: {exc}"
    else:
        report["validation_note"] = "Fewer than 3 training days: no grouped validation."

    model.fit(X, labels)
    if test_rows:
        actions = KINDS[kind]
        missed = sum(1 for m in summary.get("missed_rows") or []
                     if m["left_out"] and m["action"] in actions)
        Xt = pd.DataFrame([r["features"] for r in test_rows])[names]
        probs = model.predict_proba(Xt)[:, 1]
        report["test"] = evaluate(list(probs), [r["label"] for r in test_rows], missed)
    else:
        report["test_note"] = "No reviewed events on the left-out days yet: no test numbers."
    return model, report


def builtin_report(rows, summary):
    """The built-in (paper) exit/enter model on the same left-out days."""
    import pandas as pd

    from . import ml_scoring

    loaded = ml_scoring.load("on")
    test_rows = [r for r in rows if r["kind"] == "exit_enter" and r["left_out"]]
    if loaded is None or not test_rows:
        return None
    model, names, _kind = loaded
    probs = model.predict_proba(pd.DataFrame([r["features"] for r in test_rows])[names])[:, 1]
    missed = sum(1 for m in summary.get("missed_rows") or []
                 if m["left_out"] and m["action"] in KINDS["exit_enter"])
    return {"kind": "exit_enter", "test": evaluate(list(probs), [r["label"] for r in test_rows],
                                                   missed)}


def train_and_store(model_pk, batch_ids, left_out_text, algorithm):
    """Background job: build the set, train, evaluate, store; the
    ``CustomModel`` row goes from training to ready (or failed)."""
    import joblib

    from apps.training.models import CustomModel

    m = CustomModel.objects.get(pk=model_pk)
    kind = (m.classes or ["exit", "enter"]) == ["stay"] and "stay" or "exit_enter"
    try:
        rows, summary = dataset(m.user, batch_ids, left_out_text)
        model, report = train(rows, summary, kind, algorithm)
        report["builtin"] = builtin_report(rows, summary) if kind == "exit_enter" else None
        if model is None:
            CustomModel.objects.filter(pk=m.pk).update(status=CustomModel.Status.FAILED,
                                                       metrics=report)
            return
        buf = io.BytesIO()
        joblib.dump({"model": model, "feature_names": report["features"], "kind": kind,
                     "actions": list(KINDS[kind])}, buf)
        key = store_model(buf.getvalue(), m.user_id, kind)
        CustomModel.objects.filter(pk=m.pk).update(
            status=CustomModel.Status.READY, storage_key=key, metrics=report,
            trained_on={"batches": list(batch_ids), "left_out_days": left_out_text,
                        "events": summary["events"], "by_action": summary["by_action"]})
    except Exception as exc:  # noqa: BLE001 - shown on the page
        logger.exception("event classifier %s failed", model_pk)
        CustomModel.objects.filter(pk=m.pk).update(status=CustomModel.Status.FAILED,
                                                   metrics={"note": f"Training failed: {exc}"})
