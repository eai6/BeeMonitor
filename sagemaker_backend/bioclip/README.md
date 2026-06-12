# BioCLIP insect-ID endpoint (Phase 1)

A SageMaker **Serverless** (CPU, scale-to-zero) endpoint that zero-shot-classifies
one mover crop into a Tree-of-Life taxon. Consumed by
`beemonitor_web/apps/monitor/pipeline.py`. Design: `memory/15_…` §10.

## Contract
- `GET /ping` → 200 when the classifier is loaded.
- `POST /invocations`, `Content-Type: image/jpeg`, body = the crop JPEG →
  `{"predictions": [{"score", "common_name", "ranks": {kingdom…species}}, …]}`.

Files: `inference.py` (handlers), `serve.py` (Flask shim), `serve` (gunicorn
launcher). Image: `../Dockerfile.bioclip` (CPU torch + `pybioclip`, weights baked
in so serverless cold starts don't download).

## Deploy (CI-built, never local — see feedback_docker_in_ci_only)
1. **Build the image** — push to `main` (or run the *Build SageMaker GPU image*
   workflow); the `build-push-bioclip` job pushes to ECR repo
   `beemonitor-sm-dev-bioclip`. The ECR repo is created by `pulumi up` even with
   `deploy-bioclip=false`, so do step 2-pass-1 first if the repo doesn't exist.
2. **Create the endpoint** — in `infra/aws-sagemaker`:
   ```
   pulumi config set deploy-bioclip true
   pulumi up                # creates Model + serverless EndpointConfig + Endpoint
   pulumi stack output bioclip_endpoint_name     # -> beemonitor-sm-dev-bioclip
   ```
   Optional knobs: `bioclip-memory` (MB, default 4096), `bioclip-max-concurrency`
   (default 5), `bioclip-topk` (default 5), `bioclip-image-tag`.
3. **Point Django at it** — set `SAGEMAKER_BIOCLIP_ENDPOINT_NAME=beemonitor-sm-dev-bioclip`
   in the App Runner env (and local `.env`). Until this is set the pipeline no-ops
   and frames just ingest + display (Phase 0).
4. **Backfill** existing activities: `python manage.py classify_activities`.

## Note on `pybioclip` output shape
`predict_fn` assumes `TreeOfLifeClassifier.predict()` returns dicts with the
taxonomic-rank keys + `common_name` + `score`. If a pinned `pybioclip` version
differs, adjust the normalization in `inference.py:predict_fn` (the only coupling
point). The container is CI-built, so validate against the pinned version there.
