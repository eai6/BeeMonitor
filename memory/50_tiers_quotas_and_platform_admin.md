# 50 — Tiers, quotas, and the platform admin dashboard

**Status:** plan v3 (2026-10-21); **steps 0, 1, 4 and 5 built** (email log + test page; credit gate at every GPU entry point, UsageEvent, monthly reset, 1 credit = 1 T4 GPU-hour; sign-in history, last seen; the `/platform/` dashboard). Steps 2, 3, 6, 7, 8 not started.
Canvas: https://claude.ai/artifact/9kjvmu9SWhZQ47CzB2eAB1
The Documentation button on `/accounts/usage/` was removed separately; the real docs live on GitHub.
Your v1 comments are kept word for word in the appendix.

> **How to give feedback:** write under any `**Feedback:**` line, or edit the
> text directly. Anything marked *proposed* is a starting point, not a decision.

---

## 1. What exists today (unchanged from v1)

| Piece | Where | State |
|---|---|---|
| Tiers `free / standard / pro / enterprise` | `apps/accounts/models.py` `UserProfile.Tier`, `TIER_LIMITS` | Exists — collapses to `standard / pro` (§10) |
| Credits (1 credit ≈ 1 GPU-second) | `UserProfile.monthly_credits / used_credits` | Changing to 1 credit = 1 GPU-hour (§3) |
| Budget check before GPU work | `apps/analysis/views.py:1562` only | **Leaky** — the fix is §5 |
| Charging after GPU work | analysis poll, training, auto-label, sampling | Exists, but each place charges differently |
| Monthly reset | `reset_monthly_credits` command | Never scheduled |
| Promo codes | `Coupon`, `RedeemCouponView`, staff page `/accounts/coupons/` | Bonus credits get wiped by a reset; `upgrade_duration_days` is never enforced |
| Plan comparison table | hard-coded in `usage.html` | Out of step with `TIER_LIMITS` |
| Storage usage | `storage_used_bytes` | Never filled in — **rebuild** |
| Login tracking | Django `last_login` only | No history |

---

## 2. Tiers — decided: Standard and Pro

Free and Standard are merged: Standard is what every account starts as, and "pay as you go" just means asking for more credits. Enterprise is dropped; a large customer is a Pro account with its own limits set on the dashboard.

| | **Standard** | **Pro** |
|---|---|---|
| What it is | Every account, from sign-up | Annual subscription |
| GPU credits | **10 free / month** (reset on the 1st) + extra credits you grant on request | **Unlimited**, but tracked |
| When credits run out | New GPU work is blocked | Never |
| Raw footage | **Deleted 3 months after upload**, unless longer storage is approved | Kept while subscribed |
| Results (detections, tracks, events, crops, annotations) | Kept | Kept |
| Features | **All** | **All** |
| How they get it | Sign up | Request → you approve, with an end date (default one year) |

You can override any limit for one account (credits, retention) on its dashboard page; that replaces Enterprise.

**No per-user GPU limits (decided 2026-10-21).** The platform scales to the work for every user; jobs queue behind one platform-wide SageMaker cap (`SAGEMAKER_MAX_CONCURRENT`), which protects the endpoint and is never shown as a user limit. The per-user slot check and the "x/y slots" display were removed.

The **super admin** sits outside the tiers: unlimited, never deleted, tracked like everyone else, and has the `/platform/` dashboard.

**Units always upload.** No tier or balance ever stops footage coming in from the field. Only new GPU work is blocked.

**Feedback:**

---

## 3. Credits — 1 credit = 1 GPU-hour on the video machine, weighted by machine

### What the data says
- **YOLO video endpoint** (`ml.g4dn.xlarge`, T4, $0.736/h): the median clip takes 644 GPU-seconds for a 600-second clip (memory/48, 94 clips). **That's about 1:1, as you thought.**
  So **1 credit ≈ 1 hour of footage ≈ 6 ten-minute clips.**
- **SAM 3 endpoint** (`ml.g5.xlarge`, A10G, $1.408/h) costs 1.9× as much per hour. Training also runs on the A10G.

### Decided: credits per hour depend on the machine (rounded)

| Machine | Used for | $/h | Credits per GPU-hour |
|---|---|---|---|
| `ml.g4dn.xlarge` T4 | Video analysis, pipelines (YOLO + tracking) | 0.736 | **1** |
| `ml.g6.xlarge` L4 | (option) | 0.977 | 1.5 |
| `ml.g5.xlarge` A10G | SAM 3 auto-label, sampling, training | 1.408 | **2** |
| `ml.g6e.xlarge` L40S | (option) | 1.861 | 2.5 |

- The weights live in one table beside `pricing.py`'s instance prices and are edited by hand. Rounding keeps them easy to explain ("SAM 3 uses 2 credits an hour").
- The account page shows both: "**3.4 credits** · 2.1 GPU-hours (1.2 h video, 0.9 h SAM 3)".

### Estimating before the job starts
Each kind of work gets an estimate from **recent real runs** (rolling median over the last 200 completed jobs of that kind):

| Kind | Estimate based on |
|---|---|
| Video analysis / pipeline | GPU-seconds per second of footage × clip lengths |
| Auto-label (SAM 3) | GPU-seconds per frame × frames |
| Sampling | GPU-seconds per clip × clips |
| Training | GPU-seconds per (image × epoch) × images × epochs |

If the estimate doesn't fit in the balance, the job **doesn't start**. The message says how many credits are needed and how many are left, and offers to run only the clips that fit. If a job that was let through overruns its estimate, it still finishes; the balance can go slightly negative, which blocks the next job.

**Failed jobs:** *proposed:* not charged if it's our fault (worker crash, timeout); charged if the clip got a result.

**What this costs you:** each Standard user who uses all 10 free credits costs about **$7.40/month** in GPU time on the video machine (about the same on SAM 3, since it uses credits twice as fast). 100 active Standard users on free credits ≈ $740/month.

**Feedback:**

---

## 4. Requests — no payments on the platform

Nothing about money appears on the platform for now: no prices, no invoices, no checkout.
Users **ask**; you **approve** on the dashboard. Whatever you arrange with them happens off the platform.

A user can request:
- **More GPU credits:** an amount, plus why. Approving adds them to a balance that never expires (spent after the 10 free monthly credits).
- **An upgrade to Pro:** approved with an end date (default one year). For a bigger customer you also raise their limits.
- **Longer storage:** keep raw footage beyond 3 months, for all of it or for chosen units, until a date you set. It's kept in Glacier Instant Retrieval. See §6.

Each request has: what, how much / until when, organization, a contact email, a free-text note. You approve (optionally changing the amount or date) or decline with a note. The user sees the status on their account page.

**Pro lapsing:** a reminder appears 30 days before the end date. When it lapses, the account drops to Standard (keeping any granted credits). Footage older than 3 months then gets a 30-day notice before it is deleted, never deleted straight away.

**Promo codes** (made by you on the dashboard), for Standard users:
- *Credits:* +N credits, which never expire.
- *Tier trial:* e.g. Pro for 30 days, which now actually expires.
- Fields: code (or auto-generated), max uses, expiry date, note.

**Feedback:**

---

## 5. Enforcement — one gate, everywhere

A new `apps/accounts/quota.py`:

```python
estimate(kind, **work)                      # credits, from recent runs (§3)
check(user, credits)                        # raises QuotaExceeded(reason, needed, left, resets_on)
charge(user, kind, gpu_seconds, instance, source)   # writes a UsageEvent, updates cached totals
```

Called at **every** place that starts GPU work: analysis submit, pipeline run / batch queue,
re-run of failed clips, training & adaptation, auto-label, sampling on SageMaker.
A test lists every entry point and fails if one doesn't call the gate.

- **Monthly reset:** the 10 free credits reset on the 1st, applied on the next check after that. No cron job.
- **Balances:** `monthly` (resets), then `granted` (from requests and promo codes, never expires), spent in that order.
- **Tier expiry:** `tier_expires_at`, checked the same way.
- **Pro / super admin:** `check` always passes, but `charge` still records everything.

**Feedback:**

---

## 6. Footage retention — deleted after 3 months by default

### What storage costs (S3, us-east-1, per GB per month)

| Storage class | $/GB-month | 1 TB / month |
|---|---|---|
| S3 Standard | $0.023 | $23.55 |
| Standard-IA (clips 30–90 days old, already set up) | $0.0125 | $12.80 |
| **Glacier Instant Retrieval** (clips after 90 days, already set up) | **$0.004** | **$4.10** |
| Glacier Deep Archive | $0.00099 | $1.01 |

The raw-videos bucket already moves every clip to Glacier IR at 90 days (`infra/aws/__main__.py`, `raw-videos-lifecycle`).
The platform holds 878 GB of clips today (memory/38), which costs about **$3.50/month** once it's all in Glacier IR.
For a sense of scale: one unit recording 10-minute clips all day (~450 MB each) makes about 2 TB a month, which is about $8/month per month of footage kept in Glacier IR.

**So storage is cheap; deletion is a policy choice, not a cost saving.** It keeps the commitment simple ("we keep raw footage for 3 months") and leaves longer storage as something people ask for.

### The rule — uses the lifecycle we already have

The existing S3 lifecycle stays exactly as it is: S3 Standard → Standard-IA at 30 days → **Glacier Instant Retrieval at 90 days**. No infra change.

| A Standard user's clip at 3 months | What happens |
|---|---|
| No storage request | **Deleted at day 89**, the day *before* it would move to Glacier IR |
| Longer storage approved | Left alone: S3 moves it to Glacier IR at day 90, and it stays there until the approved date, then it's deleted |
| Request still waiting | Left alone (moves to Glacier IR as usual) until you decide |
| Request declined | Deleted from Glacier IR, after the 30 / 15 / 7-day notices |

- **Why day 89, not day 90:** Glacier IR bills every object for at least 90 days, plus a small fee to move it. Deleting a clip right after it lands in Glacier IR would pay for 3 months of storage it never uses. Deleting the day before avoids that; only clips someone asked to keep ever reach Glacier IR.
- **Kept forever, whatever happens:** results — detections, tracks, events, crops, annotation frames and boxes. Annotation projects read frames from the processed bucket, not raw clips, so labelling work survives (memory/37).
- **Pro / super admin:** nothing is deleted while the tier is active (those clips sit in Glacier IR after 90 days, as now).
- **Done by the app, per user**, not by an S3 expiry rule (that would delete Pro and super-admin footage too). A daily job deletes eligible raw files and marks the `Video` row "raw footage deleted"; it stays in Browse with its results.
- **No surprises — three emails before anything is deleted: 30, 15 and 7 days ahead.** Each one lists how many clips and GB, from which units, the deletion date, and a *Request longer storage* link. The account page shows the same notice for the whole 30 days ("142 clips will be deleted on Nov 1").
  - The notice clock starts at **day 59** after upload (30 days before the day-89 deletion), so every clip gets the full 30 / 15 / 7.
  - **Nothing is deleted unless all three emails were actually sent** (recorded per batch of clips). If an email failed, deletion for those clips waits and the next notice is retried — a broken mailer can never turn into a surprise deletion.
  - Declined storage requests and lapsed Pro accounts get the same 30 / 15 / 7 sequence, never a shorter one.
  - Your dashboard shows upcoming deletions per user and any notice that failed to send.
- **Re-running GPU work** on a deleted clip isn't possible; the run page says so and skips those clips.
- **Cost of an extension:** $0.004 per GB per month (about $4 per TB per month) — 96 GB kept for a year ≈ $4.60.

### Rollout safety (deletion can't be undone)
- Days 1–14: **dry run**. The job only reports what it *would* delete; you see the totals per user on the dashboard.
- Then: existing users get the full **30 / 15 / 7-day** sequence before their first deletion. Clips already in Glacier IR count as "older than 3 months" and are deleted at the end of that notice unless a request comes in.
- Deleted counts and bytes go into the audit log.

Questions:
- **Decided (2026-10-21): 3 months counts from upload**, the same clock the S3 lifecycle uses.

**Feedback:**

---

## 7. Data we start recording

- **`UsageEvent`** — one row per charge: user, kind, GPU-seconds, machine, credits, cost in USD, which job/run it came from, time. Everything in the dashboards is built from this.
- **`LoginEvent`** — user, time, IP, browser, success/failure, method (web / API key / device).
- **Last seen** — last web request or API call, updated at most every 5 minutes.
- **Storage per user** — recalculated when uploads land and nightly.
- **`AccountRequest`** — credit, upgrade and storage requests (§4), with their status and who approved them.
- **Retention** — per-user `raw_retention_until` (granted storage) and per-video `raw_deleted_at`.
- **`AdminAction`** — the audit log: every tier change, credit grant, promo code, suspension, view-as, with who did it.

IP and browser kept 90 days *(proposed)*.

**Feedback:**

---

## 8. The `/platform/` dashboard (super admin)

The main goal (from your comment): **manage users and see what they're doing.**

1. **Overview** — users (total, active 7d/30d, new this week), credits used this month by kind,
   GPU cost in USD this month, tier breakdown, open requests, failure rate.
2. **Users** — one row per user: name, email, organization, tier, credits this month,
   lifetime GPU-hours, storage, units, clips, last login, last seen, status.
   Sort, filter, search, export CSV.
3. **User detail** — usage over time by kind, recent runs, units, login history, requests, promo codes used.
   **Actions:** change tier (with expiry), grant credits, set limits for this user, suspend / reactivate,
   send password reset, **View as (read-only)**.
4. **Requests** — credits, upgrades, longer storage. Approve (with amount / end date) or decline.
5. **Promo codes** — create, list, deactivate (replaces `/accounts/coupons/`).
6. **Activity** — recent logins and failed logins, users who hit their limit, top users this week, admin audit log.

**View as:** a banner "Viewing as *alice* — read-only · Exit" on every page. Middleware refuses all writes while it's on. Every session is logged.

The design canvas shows all of these.

**Feedback:**

---

## 9. The account page (`/accounts/usage/`) — revised

1. **Plan card** — tier, what it includes, renewal/expiry date, and the right button: *Request credits*, *Request upgrade*, *Request longer storage*. No prices.
2. **Credits** — this month's 10 free, plus granted credits, with the reset date. Pro shows "Unlimited · 212 credits used this year".
3. **Usage chart** — credits per day this month, split by kind (analysis, pipelines, auto-label, sampling, training).
4. **Deletion notice** — Standard only: which raw clips get deleted next, when, and a *Request longer storage* button.
5. **Promo code** — stays.
6. **Job counts** — the four cards stay.
7. **Plans** — generated from `TIER_LIMITS`.
8. **Requests** — your open/past requests, if any.

The design canvas shows this.

**Feedback:**

---

## 10. Migration

- Tiers become `standard` and `pro`. Existing `free` → `standard`; `enterprise` → `pro` (with its limits copied as per-account overrides). New sign-ups start as `standard`. You → `is_superuser`.
- Existing promo codes that upgrade to `enterprise` point at `pro`.
- `HasTierPermission`: standard < pro. Since all features are on both tiers, it is barely used.
- The API's `/me` usage fields return credits in the new unit (and keep the old fields for one release).

---

## 11. Build order

Built so far (2026-10-21):
- **Step 0** — `apps/accounts/emails.py` (`send` logs every attempt to `EmailLog`, never raises); password reset sends through it (`LoggedPasswordResetForm`); superuser page `/accounts/email/` shows SES sandbox status, 7-day sent/failed counts, recent emails and a "send test email" form; SES policy (+`ses:GetAccount`) declared in `infra/aws/__main__.py` (needs `pulumi up`).
- **Account → Profile** — `/accounts/`: username (read-only), plan, first/last name, organization; change email (current password, unique, old address notified); change password (keeps this session, signs out others, notice email); "email me a reset link". The nav username opens it; Usage & credits is its second tab. New-email confirmation waits for SES production access.
- **Step 1** — `apps/accounts/quota.py`: `CREDITS_PER_GPU_HOUR` (T4 1, L4 1.5, A10G 2, L40S 2.5); estimates = median GPU-seconds per unit over the last 200 `UsageEvent`s of that kind and machine (fallbacks until 5 exist); `check` before every launch counts queued/running work; `charge` is idempotent per source. Gated: analysis one-click / new job / batch / annotated video, pipeline launches (hub + device schedules), API job create + batch submit, training, adaptation, auto-label, GPU sampling (add clips + sample). Migration 0008 backfills `UsageEvent` from completed Jobs (history only, nobody charged) and converts credit coupons from GPU-seconds to credits. Per-user concurrent-slot limits were then removed altogether (no per-user GPU limits). Until the dashboard exists, credits are granted by editing `granted_gpu_credits` in Django admin.

- **Steps 4 + 5** — `/platform/` (`apps/platform_admin/`), superusers only (everyone else gets 404), "Platform" in the nav. Overview (users, active 7/30 d, credits and AWS cost this month, storage, failure rate, credits per week by tier, users by tier, top users, recent activity); Users (search, tier/active/status filters, sort, 50 per page, CSV; per-user columns are subqueries so totals never multiply); User detail (credits left/used/lifetime/storage, credits per week by kind, recent GPU work, units, sign-ins, history) with actions: grant or remove credits (with reason), plan Standard/Pro with end date (default one year), free monthly credits override (`monthly_gpu_credits_override`), suspend/reactivate (ends their sessions, blocks sign-in and API keys; device keys still upload), send password reset; Promo codes (replaces `/accounts/coupons/`, now superuser-only, codes auto-generated, note field, plan-trial codes really end via `tier_expires_at`); Activity (audit log, sign-ins, failed sign-ins 7 d, out of credits, top users). `AdminAction` is the audit log; `LoginEvent` records web/admin/API sign-ins and failures (IP and browser pruned after 90 days); `LastSeenMiddleware` writes `last_seen_at` at most every 5 minutes, API-key calls included. `quota.expire_tier` drops an ended plan to Standard on the next check. Storage per user is computed live from `Video.file_size_bytes` rather than stored. Requests (step 6) and view-as (step 7) are not in it yet.

0. **Email that works, and that we can see working** (prerequisite for deletion notices). Found 2026-10-21: SES is still in the **sandbox** (`ProductionAccessEnabled: false`, review case 178259218200517), so SES rejects mail to anyone not verified (`MessageRejected … jqc7119@psu.edu`, 2026-10-20 19:25 and 19:32 UTC) while the reset page still says "check your email". The domain (DKIM) and the `beemonitor-ses-send` IAM policy are fine, but both were set up by hand and aren't in Pulumi. Fix: get production access; record every send (`EmailLog`: to, kind, status, SES message id, error); show failures on the dashboard; add a "send test email" button; add the SES policy to Pulumi.
1. Quota gate at every GPU entry point + estimates + `UsageEvent` + monthly reset. *(Closes the leak on its own.)*
2. Credit unit change, balances (monthly / purchased / bonus), tier expiry, new `TIER_LIMITS`.
3. Account page redesign.
4. `LoginEvent`, last seen, storage calculation.
5. `/platform/` dashboard: overview, users, user detail, actions, audit log, promo codes.
6. Requests (credits, Pro, longer storage) and approval.
7. Read-only view-as.
8. Footage deletion after 3 months (dry run first, then notices, then on).

---

## Appendix — your v1 comments, word for word

- §1 credits row: "< 1 credit = for 1 GPU-hour> The cost of GPU-hour for yolo sagemaker is differnt for SAM3 sagemaker, so we need to account for this somehow?" → §3
- §1 budget row: "yes it will be important to avoid ending work in the middle. We can use the data of the current analysis that we have done." → §3 estimates
- §1 charging row: "yes we need to also account for the other GPU work like auto-labelling and sampling which are not analysis" → §3, §5
- §1: "I think we can have it so that for each month, user get 10 credits for free for now. that is 10 hours of GPU time. This reset at the end of the month. Then we can say that the standard is a pay as you go subscrioption, where the user can pay for GPU time and so on. Then the pro will be a subcription with unlimited GPU time usage. it will be like an annual plan or something. Then enterprise will be liek quote kind of thing, where we will need to learn more about the firm to charge." → §2
- §1 promo row: "Yes I will need a feature as the super admin to be able to generate promo codes for free and standard subscrobers." → §4, §8.5
- §1 storage row: "yes we need to rebuild this" → §6, §7
- §1: "Yes lets fix the gate" → §5
- §2 tiers: "Pro <standard - pay as you go>, Pro+ <Pro - annual subscription>" → §2
- §2 over quota: "ideally we will check in advance before running a job to make sure there is enought credits" → §3, §5
- §2 view as: "I mainly want to have a dashboard to be able to manage user and track or know what they are doing" → §8
- §3: "Check the currrent data. I think with the YOLO sagemaker it is about 1:1 for clip length vs GPU time." → §3 (confirmed: 644 s per 600 s clip)
- §3: "I think for now we should not have storage limits. instead we can say that for the standard tier, where usage is pay as you go. the data will be erased or arhived after 3 months, while the data for the pro tier will be kept as payed for as part of their annual subscription." → §6
- §3: "All features should be available to all tiers" → §2
- §5: "lets go with A and C." → §4
- §5: "Yes the standard tier is pay as you go. The level up will have annual with unlimited GPU usage but tracked for now. same as the super admin" → §2, §4
- §5: "always take the data from the units. We will just block GPU usage." → §2
- §8: "i would like to see the UX/UI for this in" (workspace) → design canvas
- §8 audit log: "yes good."
- §9: "I would like to see these UX/UI changes in workspace to review there too" → design canvas
