
# BeeMonitor — AWS Migration Plan

Supersedes the Modal + Azure cloud target from `03_cloud_backend_design.md` and
`00_implementation_plan.md` (Stages 1–3). The Django web app, GPU inference, and
storage all move to AWS. The existing Modal app and Azure infra remain as
reference until the AWS stack is verified, then they are decommissioned.

**Goal.** Field Raspberry Pis upload recorded videos to S3 through an
authenticated Django API. The web app serves those videos to their owners.
Uploads automatically trigger GPU analysis on a SageMaker Async endpoint.

## Anchors — what already exists

- `beemonitor_web/` Django 5 + DRF app, gunicorn-served, with API-key auth
  (`bmk_*`, hashed, tier-throttled) at `apps/api/authentication.py`.
- `cloud/storage/azure_client.py` — current primary storage client.
- `cloud/ingestion/s3_connector.py` — S3 is only an **ingestion source** today,
  not the primary store. Keep it; it's unrelated to this migration.
- `cloud/modal_app/functions/process_video.py` — GPU pipeline on Modal L4. The
  Celery task at `apps/analysis/tasks.py:37` is not yet wired to Modal (TODO).
- `hardware/main.py` — Pi records H.264 locally to
  `/home/apis/Desktop/cameraOutput/beeHotel/`. No upload code exists.
- `infra/__main__.py` — Pulumi-Azure; will be replaced by a new Pulumi-AWS
  project, not edited in place.

## AWS reference patterns (from EcoMorph, account 495331821764)

- AWS profile `ecomorph` (SSO, `us-east-1`, `AdministratorAccess-DA`). Reuse.
- Pulumi Python projects per service, stacks `dev` / `prod`.
- GitHub Actions → AWS via OIDC role (no long-lived keys). Bootstrap script
  pattern at `EcoMorph/sagemaker_backend/bootstrap_github_oidc.sh`.
- GPU: SageMaker Async Inference on `ml.g5.xlarge`, ECR container, scale-to-zero.

---

## Phase 1 — Replace Azure Blob with S3 (direct cutover)

**Objective.** Azure Blob is removed entirely. S3 becomes the only primary
storage. **No abstraction layer or backend-switch is built** — we don't keep
Azure alive in parallel. The migration is staged across two checkpoints (1a
adds the S3 client; 1b deletes Azure and rewires every call site to S3) but
the end state is single-backend.

### Phase 1a — S3 client foundation

- `cloud/storage/s3_client.py` — boto3-backed `S3StorageClient`. No ABC.
  Uses task-role credentials on AWS; locally `AWS_PROFILE=ecomorph`.
- `cloud/storage/config.py` — add `S3Config` dataclass alongside the
  existing `StorageConfig` (Azure). The Azure config is left untouched.
- `cloud/storage/__init__.py` — export `S3StorageClient`, `S3Config`.
  Existing Azure exports kept until 1b removes Azure.
- Unit tests for the S3 client with `moto`.
- Manual S3 round-trip against the `ecomorph` profile in a real dev bucket.

Azure code paths are unchanged during 1a — nothing in production breaks.

### Phase 1b — Rip Azure out, swap to S3 everywhere

This is the high-touch refactor (~44 files, 362 Azure references):

- `beemonitor_web/apps/videos/models.py` — rename `azure_blob_path` →
  `storage_key`. Data migration copies values verbatim (no path translation
  needed; the existing strings are just keys). Same for
  `training.CustomModel.azure_model_path` → `storage_key`.
- `beemonitor_web/apps/analysis/views.py` — replace every inlined
  `BlobServiceClient.from_connection_string(...)` block with the
  `S3StorageClient`. The `_transfer_s3_to_azure` helper at lines 83–140
  is deleted (no longer needed; uploads go straight to S3 in Phase 3).
- Same swap in: `apps/analysis/video_proxy.py`, `apps/annotations/views.py`,
  `apps/videos/views.py`, `apps/training/views.py`, `apps/api/views.py`,
  `apps/api/serializers.py`, `apps/sources/views.py`,
  `apps/sources/connectors.py`, `apps/videos/tasks.py`, every management
  command under `apps/*/management/commands/`.
- `cloud/wrapper/pipeline.py`, `cloud/wrapper/model_manager.py`,
  `cloud/ingestion/direct_upload.py`, `cloud/ingestion/base_connector.py` —
  swap `AzureBlobClient` for `S3StorageClient` (reusable inside the future
  SageMaker handler).
- **Modal app archived early.** ``cloud/modal_app/`` is moved to
  ``archives/modal_app/`` during 1b.5 instead of waiting until Phase 4. The
  Django ``_spawn_modal_job`` / ``_spawn_modal_batch`` helpers are renamed
  ``_spawn_gpu_job`` / ``_spawn_gpu_batch`` and stubbed — they mark jobs
  ``failed`` with a clear "Phase 4 pending" message. Phase 4 fills them in.
- Delete: `cloud/storage/azure_client.py`,
  `cloud/storage/container_setup.py` (S3 prefixes don't need setup),
  `cloud/ingestion/azure_connector.py`, `StorageConfig` from
  `cloud/storage/config.py`. Drop `azure-storage-blob` and `azure-identity`
  from `beemonitor_web/requirements/base.txt` and
  `cloud/requirements-cloud.txt`.
- `config/settings/base.py` — replace `AZURE_STORAGE_*` with
  `AWS_S3_BUCKET`, `AWS_REGION`. Delete `AZURE_STORAGE_CONNECTION_STRING`,
  `AZURE_STORAGE_CONTAINER`.

**Out of scope for Phase 1.** No infra-as-code yet, no deploy, no changes to
the Modal pipeline beyond the storage swap. Pi still records locally.

**Validation.**
- `python manage.py shell` → instantiate `S3StorageClient`, upload a 10 MB
  file to a dev bucket, fetch a presigned URL, download via the URL, delete.
- Unit tests pass with `moto` mocking S3.
- Existing Django test suite passes after the migration (tests that hardcode
  `azure_blob_path=...` get renamed).
- A pre-migration video record's `storage_key` matches its previous
  `azure_blob_path` value byte-for-byte.

## Phase 2 — AWS infrastructure (Pulumi)

**Objective.** A `dev` stack of the Django app running on AWS, talking to RDS
Postgres, Redis, and the S3 bucket from Phase 1. No GPU yet.

**Deliverables.** New Pulumi-Python project at `infra/aws/` (the existing
`infra/__main__.py` Azure stack is left untouched as a reference until cutover):
- S3 bucket: versioning, lifecycle (raw videos → IA after 30d, Glacier after
  90d), block-public-access on, server-side encryption.
- RDS Postgres 16 (`db.t4g.micro` for dev), single-AZ.
- ElastiCache Redis (`cache.t4g.micro`) **or** Upstash if cheaper for the
  expected scale — decide during Phase 2.
- ECR repo `beemonitor-web` for the Django image.
- ECS Fargate service behind an ALB + ACM cert. App Runner is rejected: we
  need ALB listener rules for the future Pi upload path and SageMaker callbacks.
- Secrets Manager: Django `SECRET_KEY`, DB password, JWT/API hash pepper.
- IAM task role with least-privilege S3 + Secrets access.
- GitHub Actions OIDC role `beemonitor-github-actions` (mirror EcoMorph's
  `bootstrap_github_oidc.sh`). Repo: `Team-Insect-Net/BeeMonitor_eai6`.
- Route53 record + ACM cert for `dev.beemonitor.<domain>` — domain TBD.

**Validation.**
- `pulumi up` succeeds on the `dev` stack.
- A CI build pushes the Django image to ECR, the Fargate service rolls,
  `/health` returns 200 over HTTPS.
- A logged-in user can list their existing videos (the S3 bucket starts empty
  in dev; seed a couple of test objects via `aws s3 cp`).

## Phase 3 — Raspberry Pi → S3 via authenticated API

**Objective.** Pis upload recorded videos to S3 without ever holding AWS
credentials. Every uploaded object's S3 key encodes the owning user + device,
so Phase 5 scoping is enforced by the key layout itself.

**Deliverables.**
- `beemonitor_web/apps/devices/` (new app): `Device` model with `owner` (FK
  to User), `name`, `device_key` (hashed `bmk_device_*`), `last_seen_at`.
  Admin can issue/revoke keys.
- New endpoints:
  - `POST /api/v1/uploads/initiate` — authenticated by device key. Body:
    `filename`, `size_bytes`, `recorded_at`, `content_type`. Returns
    a presigned multipart S3 upload (or single PUT if size < 100 MB), with
    key `users/<user_id>/devices/<device_id>/<yyyy>/<mm>/<dd>/<uuid>.<ext>`.
  - `POST /api/v1/uploads/complete` — Pi confirms upload + multipart parts.
    Django finalizes the multipart upload, creates a `Video` row with
    `storage_key`, `device`, `owner`, and enqueues analysis (Phase 4 wires
    this; until then it just creates the row).
- `hardware/uploader.py` — new module. Watches the recording dir, on new
  `.mp4` calls `/initiate`, performs the multipart PUTs with retries +
  exponential backoff, calls `/complete`, moves the file to a local
  `uploaded/` archive. Runs as a **separate systemd service** from
  `main.py` so a network outage cannot stop recording.
- `hardware/systemd/beemonitor-uploader.service` — unit file. Logs to
  journald. Restart-on-failure with a backoff cap.
- ~~Bucket policy: deny PutObject outside the user/device prefix.~~
  Dropped after analysis: the Pi receives a presigned URL bound to one
  exact ``(bucket, key, content-type)``; S3 itself rejects any PUT under
  a different key. The storage key is generated server-side from the
  device's owner_id + id, and ``/uploads/complete`` re-verifies the
  prefix before creating the ``Video``. A bucket policy would also break
  the legacy ``_upload_to_storage`` path used by the Django web upload,
  which uses ``{user_pk}/{upload_id}/...`` keys.

**Validation.**
- A real Pi with a freshly issued device key uploads a 200 MB recording
  end-to-end. The `Video` row appears in the dashboard for the device owner
  and nobody else.
- Power-cycling the Pi mid-upload: the uploader resumes and completes.

## Phase 4 — Automatic GPU analysis on upload

**Objective.** Every uploaded video triggers BeeMonitor's detection + tracking
pipeline on a SageMaker Async endpoint and stores results back in S3 + the
`AnalysisJob` table.

**Approach decision.**
- **Trigger.** `uploads/complete` enqueues a Celery task that invokes the
  SageMaker Async endpoint. S3 EventBridge is *not* used in Phase 4 — the
  API completion event already carries owner + device context that the S3
  event lacks. Revisit if a non-API ingest path is added later.
- **Fill in the spawn stubs.** `apps/analysis/views.py::_spawn_gpu_job`
  and `_spawn_gpu_batch` are currently stubs that fail jobs with
  "Phase 4 pending". Replace the stub bodies with the actual SageMaker
  Async invoke + result-polling Celery task.
- **Inference container.** Build the SageMaker handler from
  `cloud/wrapper/pipeline.py` (already S3-native) — mirror EcoMorph's
  `sagemaker_backend/inference.py`. Image lives in ECR. The archived
  `archives/modal_app/` is read-only reference for what the function
  signatures used to be; don't take a runtime dep on it.
- **Pulumi project.** New `infra/aws-sagemaker/` (separate from the web
  stack so the GPU stack scales independently). Models: ECR repo, S3 results
  prefix, SageMaker model + endpoint config + Async endpoint, scale-to-zero
  autoscaling. Mirror `EcoMorph/sagemaker_backend/infra/__main__.py`.
- **Result handling.** SageMaker writes the JSON result to
  `s3://<bucket>/results/<job_id>.json`. A second Celery task polls (or
  receives SNS notification) and updates the `AnalysisJob` row.

**Validation.**
- A fresh Pi upload triggers a job that completes within 2× the video
  duration on a 1-minute test clip. Results appear in the dashboard.
- The endpoint scales to zero after the idle window; the next upload cold-
  starts it without the API call failing.

## Phase 5 — User and role scoping

**Objective.** Every read path enforces ownership. A user only sees their own
videos, devices, and analyses. Support staff with an `admin` flag can see all.

**Deliverables.**
- Queryset mixins / DRF permissions filtering by `request.user` everywhere
  videos, devices, analyses, and dashboards are listed.
- Audit pass: every view in `apps/videos/`, `apps/analysis/`,
  `apps/dashboard/`, `apps/devices/`. Add tests that a second user gets 404
  on the first user's resources.
- Role flag on User (`is_support`); bypass for support reads only — writes
  still go through ownership checks.
- Presigned-URL endpoint never returns a URL for an object the user doesn't
  own (defence in depth even though the S3 prefix layout already prevents
  cross-tenant access).

**Validation.**
- Two test users with different devices; each sees only their own videos.
  Verified in the UI and via the API.

---

## Cutover and decommissioning

- After Phase 2 ships to `prod` and one week of stable traffic, point DNS
  away from any Azure resources and `pulumi destroy` the Azure stack
  (`infra/__main__.py`) under confirmation.
- The Modal app has already been archived to ``archives/modal_app/`` during
  Phase 1b. After Phase 4 ships and is stable for two weeks, the archived
  directory can be deleted (no live system depends on it).

## Risks worth naming

- **Long-running uploads on flaky cellular.** Multipart + resume is in
  Phase 3 for this reason; a single PUT would fail too often.
- **SageMaker Async cold start.** ~2 minutes for the first request after
  scale-to-zero. Acceptable for batched bee video analysis; surface job
  state clearly in the UI so users don't refresh-spam.
- **SageMaker Async 1h per-invocation cap.** Hard limit. BeeMonitor's
  typical 5–15 min videos process in 10–30 min on a g4dn — fits. Long
  recordings (≥30–45 min source) risk hitting the cap. Mitigations when
  it bites: (a) chunk the video client-side on the Pi before upload, or
  (b) move long videos to SageMaker Batch Transform (no 1h cap). The
  gunicorn worker timeout (14400 s) is intentionally well above the SM
  cap so SageMaker fails first with a clean ``.failure`` object the
  poller reads — gunicorn killing the worker mid-stream is hard to
  observe.
- **Cost.** RDS + Fargate + Redis floor is ~$60/mo before traffic. Document
  before the first deploy so the Penn State billing path is in place.
- **Two-cloud period.** Phase 1 ships with Azure still primary so nothing
  breaks. Phase 2 flips the switch. Don't try to do both at once.
