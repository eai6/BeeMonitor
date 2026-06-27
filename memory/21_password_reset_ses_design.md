# 21 — Self-serve password reset (email via AWS SES + django-anymail)

## Context

The web app (`beemonitor_web`, Django ≥5.0 on AWS App Runner) has login / logout /
register but **no password reset** — a user who forgets their password is locked
out. We want the standard Django "forgot password" flow: enter email → receive a
reset link → set a new password. This also unblocks testing device-share video
access across multiple accounts.

The app already uses boto3 with the IAM credential chain (App Runner task role in
prod; `AWS_PROFILE=ecomorph` SSO locally — `cloud/storage/s3_client.py:8-9`). There
is currently **no `EMAIL_*` config anywhere** — that's the gap.

## Decisions (user-confirmed + research-backed)

- **Delivery:** AWS SES via the **django-anymail** package (SES v2 API backend). It
  reuses the existing boto3 IAM credential chain — no SMTP credentials to mint or
  store. (Deep-research recommendation; verified against AWS + Anymail docs.)
- **From identity:** domain `beemonitor.edwardamoah.com` (subdomain of a domain the
  user controls), e.g. `no-reply@beemonitor.edwardamoah.com`. Kept env-driven
  (`DEFAULT_FROM_EMAIL`) so it can change later. SES **cannot** send from
  `awsapprunner.com`.
- **Security:** rely on Django's built-in reset views (they don't leak whether an
  email exists — always render the same "done" page). Backend must **fail
  gracefully** on SES errors, because a consistent send-failure (500) is itself a
  user-enumeration signal. Reset links are `https://` automatically because
  production sets `SECURE_PROXY_SSL_HEADER` (`config/settings/production.py:24`) so
  `request.is_secure()` is True behind App Runner's TLS termination.

## Implementation

### 1. Dependency — `requirements/base.txt`
Add `django-anymail[amazon-ses]>=11.0` (boto3 already pinned at `boto3>=1.28.0`).

### 2. Settings — `config/settings/base.py`
- Add `"anymail"` to `INSTALLED_APPS`.
- `ANYMAIL = {"AMAZON_SES_CLIENT_PARAMS": {"region_name": AWS_REGION}}` (reuse the
  existing `AWS_REGION`).
- `DEFAULT_FROM_EMAIL = os.environ.get("DEFAULT_FROM_EMAIL", "no-reply@beemonitor.edwardamoah.com")`
  and `SERVER_EMAIL = DEFAULT_FROM_EMAIL`.
- `EMAIL_BACKEND = os.environ.get("EMAIL_BACKEND", "django.core.mail.backends.console.EmailBackend")`
  — **console by default** so local dev needs no SES; overridden in prod.
- `PASSWORD_RESET_TIMEOUT = int(os.environ.get("PASSWORD_RESET_TIMEOUT", 60 * 60 * 24))`
  (1 day; tunable).

### 3. Settings — `config/settings/production.py`
Default the backend to SES there:
`EMAIL_BACKEND = os.environ.get("EMAIL_BACKEND", "anymail.backends.amazon_ses.EmailBackend")`.

### 4. URLs — `apps/accounts/urls.py`
Add four routes using Django's **built-in** views (no custom view code), each with a
custom `template_name` and explicit `success_url`:
- `password_reset/` → `PasswordResetView` (template + `email_template_name` +
  `subject_template_name`, `success_url=reverse_lazy("accounts:password_reset_done")`)
- `password_reset/done/` → `PasswordResetDoneView`
- `reset/<uidb64>/<token>/` → `PasswordResetConfirmView`
  (`success_url=reverse_lazy("accounts:password_reset_complete")`)
- `reset/done/` → `PasswordResetCompleteView`

### 5. Templates — `apps/accounts/templates/accounts/`
Four page templates styled to match `login.html` (extend `base.html`, `max-w-md`
card, amber-600 buttons): `password_reset_form.html`, `password_reset_done.html`,
`password_reset_confirm.html`, `password_reset_complete.html`. Plus two email
templates: `password_reset_email.html` (body, uses
`{{ protocol }}://{{ domain }}{% url 'accounts:password_reset_confirm' uidb64=uid token=token %}`)
and `password_reset_subject.txt` (one line, no newline).

### 6. Login link — `apps/accounts/templates/accounts/login.html`
Add a "Forgot your password?" link → `{% url 'accounts:password_reset' %}` near the
existing "Register" link.

### 7. Tests — `apps/accounts/tests.py` (new)
Following `apps/monitor/tests.py` patterns (`TestCase`, `create_user`,
`override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")`):
- requesting reset for a known email renders done + sends one mail containing a link;
- unknown email → same done page, **no** mail, no error (enumeration guard);
- confirm flow with a valid uid/token sets a new password and lets the user log in.

## AWS / ops prerequisites (separate from code; user/ops to do)

1. **Verify the domain identity** `beemonitor.edwardamoah.com` in SES (Easy DKIM,
   2048-bit) — add the CNAME/DKIM (and SPF + a DMARC) records to DNS. Domain identity
   covers all addresses/subdomains; SES uses strict DKIM alignment by default.
2. **Request SES production access** (mail type: *Transactional*). Sandbox limits:
   200 msgs/24h, 1 msg/sec, and sends **only to verified recipients** — so until
   production access is granted, only verified test addresses receive reset mail.
   Initial AWS Support response ~24h.
3. **IAM**: grant the App Runner role `ses:SendEmail` and `ses:SendRawEmail`.
4. **Env vars** on App Runner: `EMAIL_BACKEND` (or rely on prod default),
   `DEFAULT_FROM_EMAIL=no-reply@beemonitor.edwardamoah.com`, optional
   `PASSWORD_RESET_TIMEOUT`.

## Verification

- **Unit:** `python manage.py test apps.accounts` (locmem backend; no AWS needed).
- **Local manual:** run with the console `EMAIL_BACKEND`, hit `/accounts/password_reset/`,
  copy the link printed to the console, complete the reset, log in with the new password.
- **Prod smoke (after SES identity verified):** trigger a reset to a verified address
  (while sandboxed) and confirm delivery; confirm the link is `https://` and within the
  timeout window.

## Notes / out of scope
- Optional later hardening: rate-limit `/accounts/password_reset/` (e.g. django-axes
  or a throttle) to slow reset-spam — not required for this change.
- DKIM/DMARC DNS and SES production-access approval are ops steps; reset emails won't
  deliver to arbitrary users until both are done.

## Research provenance
- Backend choice (SES v2 API backend over SMTP; django-anymail vs django-ses) and
  SES sandbox/identity/IAM facts come from a deep-research pass verified against
  primary AWS SES docs and the Anymail docs. django-anymail chosen as the
  actively-maintained option; django-ses is a drop-in alternative but has a stated
  maintenance caveat.
