"""
STEM lesson packs (Phase 3c) — the education layer over the pipeline builder.

A lesson is an explainable, pre-built pipeline: it pairs one of the seed templates
with learning objectives, plain-language explanation of what each analysis does and
why, and guided questions for students. The lesson page shows the pipeline's steps,
lets a learner "Start" it (clones the template into their own editable pipeline), and
they can then run it or export it as a Colab notebook (Phase 3b).

Lessons are content, not models — edit this file to add or refine them. Each
``template`` must match the title of a seeded template
(``manage.py seed_pipeline_templates``).
"""

LESSONS = {
    "foraging-trips": {
        "title": "How bees forage: measuring trips",
        "level": "Beginner",
        "summary": "Track bees at a nest and measure their foraging trips — the "
                   "round-trips out to collect pollen and back.",
        "template": "Foraging trips",
        "objectives": [
            "Explain what a foraging trip is and why it matters for pollination.",
            "Follow how a video becomes tracks, then events, then trips.",
            "Read a trips table and reason about foraging effort.",
        ],
        "sections": [
            {"heading": "The question",
             "body": "Solitary bees leave their nest to forage and return to provision "
                     "their brood. Counting and timing those trips tells us how hard bees "
                     "are working and how healthy the habitat is."},
            {"heading": "How the pipeline answers it",
             "body": "We locate the nest tubes (ROI), track every bee across the video, "
                     "then detect when a track exits and later re-enters a tube. Each "
                     "exit→entry pair is one foraging trip, with a duration."},
            {"heading": "What to look at",
             "body": "The trips table shows how many trips happened and their average "
                     "duration. Longer or fewer trips can mean forage is farther away."},
        ],
        "questions": [
            "Why do we detect the nest tubes before tracking the bees?",
            "If average trip duration doubled, what might have changed in the habitat?",
            "How could a mis-tracked bee create a false trip? How would you guard against it?",
        ],
    },
    "flower-visits": {
        "title": "Pollinator visits to a flower",
        "level": "Beginner",
        "summary": "Draw a region around a flower and count how many bees visit it, "
                   "how long they stay, and which species show up.",
        "template": "Flower / ROI visitation",
        "objectives": [
            "Define a region of interest (ROI) and a 'visit'.",
            "Distinguish visitation rate from dwell time.",
            "Connect visitation to species identity.",
        ],
        "sections": [
            {"heading": "The question",
             "body": "Which flowers get visited most, and by whom? Visitation is a core "
                     "measure of pollination services."},
            {"heading": "How the pipeline answers it",
             "body": "You draw an ROI over the flower. We track bees, count each track "
                     "that enters the ROI as a visit (with dwell time), and identify the "
                     "species of the visitors."},
            {"heading": "What to look at",
             "body": "Compare visits vs. dwell time: many short visits and a few long ones "
                     "mean different things for pollen transfer."},
        ],
        "questions": [
            "Where exactly should the ROI edge sit — on the petals, or wider? Why?",
            "Two flowers get the same number of visits but very different dwell times. "
            "Which likely gets more pollination?",
            "Why might species identity change the meaning of a visit?",
        ],
    },
    "individual-ids": {
        "title": "Following individuals: marking & re-ID",
        "level": "Intermediate",
        "summary": "Read per-bee colour, QR, or number tags so each trajectory belongs "
                   "to a known individual.",
        "template": "Individual bee IDs",
        "objectives": [
            "Explain why individual identity (not just counts) unlocks new questions.",
            "Understand how a marker turns a track into a named individual.",
            "Reason about identification errors and confidence.",
        ],
        "sections": [
            {"heading": "The question",
             "body": "Counts tell you how many bees; identity tells you which bee. With "
                     "individual IDs you can measure how often one bee forages, or whether "
                     "individuals specialise on particular flowers."},
            {"heading": "How the pipeline answers it",
             "body": "Bees carry tags (paint marks, number tags, or QR codes). The tracker "
                     "reads a tag per detection; we take each track's most-frequent tag as "
                     "its identity, with a confidence."},
            {"heading": "What to look at",
             "body": "Unique markers vs. identified tracks: a gap means some tracks never "
                     "got a confident read."},
        ],
        "questions": [
            "What questions become answerable once bees are individually identified?",
            "A track's tag is read as 'blue-07' in 18 frames and 'blue-01' in 2. Which "
            "identity do we assign, and why take the mode?",
            "How would motion blur affect QR vs. colour tags differently?",
        ],
    },
    "colony-activity": {
        "title": "Inside the nest: colony activity over time",
        "level": "Intermediate",
        "summary": "Measure how busy a nest is through the day by binning bee activity "
                   "into a time series.",
        "template": "Colony activity",
        "objectives": [
            "Turn per-frame tracks into a time series.",
            "Interpret occupancy vs. detection-rate metrics.",
            "Relate activity peaks to biological rhythms.",
        ],
        "sections": [
            {"heading": "The question",
             "body": "When is a nest most active? Activity rhythms reveal foraging bouts, "
                     "temperature responses, and colony health."},
            {"heading": "How the pipeline answers it",
             "body": "We track bees, then bin the video into short time windows and count "
                     "how many distinct bees appear in each (occupancy), producing a curve "
                     "over time."},
            {"heading": "What to look at",
             "body": "Peaks and troughs in the curve. A single tall peak vs. a broad plateau "
                     "describe very different activity patterns."},
        ],
        "questions": [
            "Why bin by time instead of counting every frame?",
            "Occupancy and detection-rate can disagree — when, and what does each capture?",
            "What environmental variable would you overlay on the activity curve, and why?",
        ],
    },
}


def get_lesson(slug):
    return LESSONS.get(slug)


def list_lessons():
    return [{"slug": slug, **lesson} for slug, lesson in LESSONS.items()]
