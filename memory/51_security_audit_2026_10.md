# 51 — Security audit, October 2026

**Status:** fixes built and pushed 2026-10-21. Three read-only audits (git history, AWS + CI, the Django app) after the superuser bootstrap; the repo is public, so every finding assumed the attacker has the source.

## Found and fixed

| Severity | Finding | Fix |
|---|---|---|
| Critical | `POST /api/v1/auth/reset-password/` set any account's password from an email alone (no token, no login). Access logs since May 23 show it was never called. | Endpoint deleted (`5d1f653`). |
| High | Old admin password in public git history (README, entrypoint, Mar 2026). | Owner changes the password; history not rewritten (forks already have it). |
| High | Stored XSS: JSON written into `<script>` with `|safe`; JS building HTML from unit names, Wi-Fi SSIDs, class/project names; template values in inline `onclick`. | `|js` builtin filter (`config/template_filters.py`), `esc()` in the JS, `|escapejs` in handlers, Markdown fallbacks escaped. |
| High | Credit gate skipped by the pipeline run API, re-run and swap-analyzer; parallel launches each saw the whole balance. | `engine.start_run` checks unless the caller did (`checked=True`); `quota.check` takes a per-user row lock and writes a 60 s `CreditHold`. |
| Medium | API accepted raw `custom_*_model_path`: any tenant's private model ran. | `api.serializers.resolve_custom_models`: by id or path, only `CustomModel.usable(user)`. |
| Medium | Superuser bootstrap trusted an unverified email; the secret stayed on App Runner. | Mechanism removed (command, entrypoint line, env). Promote in Django admin. |
| Medium | Reset links built from the Host header; CSRF trusted every `*.awsapprunner.com`. | `SITE_DOMAIN` fixes the link domain; CSRF trusts the custom domain only. `ALLOWED_HOSTS` stays `*` (App Runner health check). |
| Medium | No brute-force limit. | `ThrottledModelBackend`: 10 failures / username or 30 / address in 15 min locks sign-in (web, admin, API), counted from `LoginEvent`. Reset email 5/day/address, promo codes 10/hour/account (`Attempt`). |
| Medium | `SECRET_KEY` fell back to a value in the repo. | Production refuses to start without one. |
| Medium | SES policy allowed sending as the personal verified addresses; addresses logged in never-expiring logs. | `ses:FromAddress` condition; failures log the EmailLog id, not the address. |
| Medium | Unpinned dependencies and CDN scripts. | `requirements/base.lock` (hashes, Linux, nothing newer than two weeks; CI tests install it too); CDN scripts pinned with SRI. |
| Low | API: weak passwords accepted, a video could point at another user's DataSource, API-created credentials stored in plain JSON (and its create 500'd echoing them), upload names kept `../`, GET logout. | Fixed each; `sources/crypto.py` encrypts and still reads old plain rows. |
| Low | Infra: RDS deletion protection off, no TLS-only bucket policy, localhost CORS in prod, `refs/tags/*` OIDC, unused ECS roles that could read secrets, ECR pull on `*`, actions unpinned, minisign tarballs unchecked, base image CVEs (critical: `tiff`). | All in Pulumi / workflows / Dockerfile; three unused repo secrets (Azure, Modal) deleted. |

## Decided not to change
- **Unit re-binding** with an enrollment token + card id: by design (re-flashed cards recover this way). A leaked token is revoked instead.
- **SageMaker network isolation** stays off: containers download weights at start.
- **Account-level S3 Block Public Access**: the AWS account is shared with other projects.
- **ECR immutable tags / deploy by digest**: App Runner auto-deploys `:latest`; a bigger change for later.
- **Org read-only roles** (Kion, Tenable, …) can read the buckets: institutional, outside this repo.

## Owner to do
1. Change the `edwardamoah`/`eai6` password.
2. `pulumi config rm superusers`, then `pulumi up` on `infra/aws` (after the app deploy with `SITE_DOMAIN`), then on `infra/aws-sagemaker`. Then send a test email from `/accounts/email/` to confirm the SES condition.
3. Pulumi secrets are passphrase-encrypted in a public repo: move to KMS (`pulumi stack change-secrets-provider awskms://…`) or make sure the passphrase is long and random; rotate the Anthropic key if not.
4. CloudWatch log retention (App Runner, SageMaker groups never expire): decide a period; deleting old logs can't be undone, and they were how the reset endpoint was cleared.
5. Optional: GitHub rulesets for `main` and `v*` tags.
