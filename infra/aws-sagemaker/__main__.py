"""
Pulumi IaC for BeeMonitor GPU inference on AWS SageMaker Async Inference.

A SEPARATE Pulumi project from the web stack (infra/aws/) so the GPU backend
ships and scales independently of the Django web tier. See Phase 4 of
``memory/09_aws_migration_plan.md``.

Deploys (in order, gated by ``deploy-endpoint``):

  Always:
    - ECR repository ``beemonitor-sm-{env}`` for the GPU inference image.
    - 2 S3 buckets:
        beemonitor-sm-{env}-input-{acct}  (async request payloads)
        beemonitor-sm-{env}-output-{acct} (async result JSON, dropped by SM)
      Both private with lifecycle expiry — these are transient.
    - IAM SageMaker execution role (assumed by the inference container).
    - IAM managed policy ``beemonitor-sm-{env}-django-invoke`` — attached to
      the App Runner instance role in the web stack so Django can invoke
      the endpoint + write input payloads + read async output.

  When ``deploy-endpoint=true``:
    - SageMaker Model / EndpointConfig (Async) / Endpoint.
    - Application Auto Scaling target+policy. MinCapacity=0 → the endpoint
      scales to zero after idle; the property that makes it serverless on
      cost.

Two-pass deploy
---------------
The Endpoint resource pulls the inference image, so the image must already
exist in ECR before ``deploy-endpoint`` can flip on:

  1. ``pulumi up`` with deploy-endpoint=false  → ECR + S3 + IAM only.
  2. CI workflow builds and pushes the GPU image.
  3. Flip deploy-endpoint=true, ``pulumi up`` → Model + Endpoint + autoscaling.
"""

import json

import pulumi
import pulumi_aws as aws


# ---------------------------------------------------------------------------
# Config + identity
# ---------------------------------------------------------------------------

config = pulumi.Config()
env = config.get("environment") or "dev"
instance_type = config.get("instance-type") or "ml.g4dn.xlarge"
image_tag = config.get("image-tag") or "latest"
deploy_endpoint = config.get_bool("deploy-endpoint") or False
deploy_bioclip = config.get_bool("deploy-bioclip") or False
max_capacity = config.get_int("max-capacity") or 2
# SAM 3 auto-labeler (GPU async, scale-to-zero). GPU + large weights, so default to
# g5.xlarge (24 GB A10G). Only runs at label time, so max-capacity stays small.
deploy_sam3 = config.get_bool("deploy-sam3") or False
sam3_instance_type = config.get("sam3-instance-type") or "ml.g5.xlarge"
sam3_max_capacity = config.get_int("sam3-max-capacity") or 2

prefix = f"beemonitor-sm-{env}"
ECR_REPO_NAME = f"{prefix}"  # must match the CI workflow
VARIANT_NAME = "AllTraffic"

account_id = aws.get_caller_identity().account_id
region = aws.get_region().name


# ---------------------------------------------------------------------------
# ECR — holds the GPU inference image (CI-built, never local)
# ---------------------------------------------------------------------------

ecr_repo = aws.ecr.Repository(
    "inference-repo",
    name=ECR_REPO_NAME,
    image_tag_mutability="MUTABLE",  # CI re-pushes :latest on main
    image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(
        scan_on_push=True,
    ),
    force_delete=False,
)

aws.ecr.LifecyclePolicy(
    "inference-repo-lifecycle",
    repository=ecr_repo.name,
    policy=json.dumps({
        "rules": [{
            "rulePriority": 1,
            "description": "Expire untagged images after 14 days",
            "selection": {
                "tagStatus": "untagged",
                "countType": "sinceImagePushed",
                "countUnit": "days",
                "countNumber": 14,
            },
            "action": {"type": "expire"},
        }],
    }),
)

# ECR for the BioCLIP insect-ID image (CPU, CI-built). Separate repo from the
# video image so the two build/deploy independently.
BIOCLIP_ECR_REPO_NAME = f"{prefix}-bioclip"  # must match the CI workflow
bioclip_ecr_repo = aws.ecr.Repository(
    "bioclip-repo",
    name=BIOCLIP_ECR_REPO_NAME,
    image_tag_mutability="MUTABLE",
    image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(
        scan_on_push=True,
    ),
    force_delete=False,
)

# ECR for the SAM 3 auto-labeler image (GPU, CI-built). Its own repo so it
# builds/deploys independently of the video + bioclip images.
SAM3_ECR_REPO_NAME = f"{prefix}-sam3"  # must match the CI workflow
sam3_ecr_repo = aws.ecr.Repository(
    "sam3-repo",
    name=SAM3_ECR_REPO_NAME,
    image_tag_mutability="MUTABLE",
    image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(
        scan_on_push=True,
    ),
    force_delete=False,
)


# ---------------------------------------------------------------------------
# S3 — async input + output buckets (transient, short lifecycle)
# ---------------------------------------------------------------------------

def _make_bucket(logical: str, purpose: str, expire_days: int) -> aws.s3.BucketV2:
    bucket = aws.s3.BucketV2(
        logical,
        bucket=f"{prefix}-{purpose}-{account_id}",
    )
    aws.s3.BucketPublicAccessBlock(
        f"{logical}-pab",
        bucket=bucket.id,
        block_public_acls=True,
        block_public_policy=True,
        ignore_public_acls=True,
        restrict_public_buckets=True,
    )
    aws.s3.BucketServerSideEncryptionConfigurationV2(
        f"{logical}-sse",
        bucket=bucket.id,
        rules=[aws.s3.BucketServerSideEncryptionConfigurationV2RuleArgs(
            apply_server_side_encryption_by_default=aws.s3
                .BucketServerSideEncryptionConfigurationV2RuleApplyServerSideEncryptionByDefaultArgs(
                    sse_algorithm="AES256",
                ),
            bucket_key_enabled=True,
        )],
    )
    aws.s3.BucketLifecycleConfigurationV2(
        f"{logical}-lifecycle",
        bucket=bucket.id,
        rules=[aws.s3.BucketLifecycleConfigurationV2RuleArgs(
            id=f"expire-{purpose}",
            status="Enabled",
            filter=aws.s3.BucketLifecycleConfigurationV2RuleFilterArgs(prefix=""),
            expiration=aws.s3.BucketLifecycleConfigurationV2RuleExpirationArgs(
                days=expire_days,
            ),
        )],
    )
    return bucket


# Input + output are transient — request payloads and result JSON only.
# The actual video bytes live in the web stack's `raw-videos` bucket;
# the actual analysis outputs (CSVs, annotated videos) live in `processed`.
input_bucket = _make_bucket("input-bucket", "input", expire_days=7)
output_bucket = _make_bucket("output-bucket", "output", expire_days=7)


# ---------------------------------------------------------------------------
# IAM — SageMaker execution role (runtime principal of the container)
# ---------------------------------------------------------------------------
# Least-privilege:
#   - Read sm-input bucket (request payloads).
#   - Write sm-output bucket (async result JSON dropped by SM).
#   - Read raw-videos + write processed (the CloudPipeline pulls + writes).
#   - Read models bucket (ensure_models / ensure_custom_model).
#   - Pull from ECR. Write CloudWatch.

web_raw_videos_arn = f"arn:aws:s3:::beemonitor-{env}-raw-videos-{account_id}"
web_processed_arn = f"arn:aws:s3:::beemonitor-{env}-processed-{account_id}"
web_models_arn = f"arn:aws:s3:::beemonitor-{env}-models-{account_id}"

sagemaker_role = aws.iam.Role(
    "sagemaker-exec-role",
    name=f"{prefix}-exec-role",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }),
)

sagemaker_role_policy = aws.iam.RolePolicy(
    "sagemaker-exec-policy",
    role=sagemaker_role.id,
    policy=pulumi.Output.all(
        input_bucket.arn, output_bucket.arn,
    ).apply(lambda arns: json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ReadAsyncInput",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [arns[0], f"{arns[0]}/*"],
            },
            {
                "Sid": "WriteAsyncOutput",
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
                "Resource": [arns[1], f"{arns[1]}/*"],
            },
            {
                "Sid": "ReadRawVideos",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [web_raw_videos_arn, f"{web_raw_videos_arn}/*"],
            },
            {
                "Sid": "WriteProcessed",
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
                "Resource": [web_processed_arn, f"{web_processed_arn}/*"],
            },
            {
                "Sid": "ReadModels",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [web_models_arn, f"{web_models_arn}/*"],
            },
            {
                "Sid": "ECRPull",
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                ],
                "Resource": "*",
            },
            {
                "Sid": "CloudWatch",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                    "cloudwatch:PutMetricData",
                ],
                "Resource": "*",
            },
        ],
    })),
)


# ---------------------------------------------------------------------------
# IAM — Django invoke policy (attached to App Runner role in the web stack)
# ---------------------------------------------------------------------------
# Standalone managed policy so the web stack can attach by ARN without
# pulumi StackReference. The web stack reads:
#   arn:aws:iam::<account>:policy/beemonitor-sm-<env>-django-invoke

django_invoke_policy = aws.iam.Policy(
    "django-invoke-policy",
    name=f"{prefix}-django-invoke",
    description="Django web app: invoke the BeeMonitor SageMaker async endpoint + S3 I/O",
    policy=pulumi.Output.all(
        input_bucket.arn, output_bucket.arn,
    ).apply(lambda arns: json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeAsyncEndpoint",
                "Effect": "Allow",
                "Action": "sagemaker:InvokeEndpointAsync",
                "Resource": (
                    f"arn:aws:sagemaker:{region}:{account_id}:endpoint/{prefix}"
                ),
            },
            {
                "Sid": "InvokeBioclipEndpoint",
                "Effect": "Allow",
                "Action": "sagemaker:InvokeEndpoint",
                "Resource": (
                    f"arn:aws:sagemaker:{region}:{account_id}:endpoint/{prefix}-bioclip"
                ),
            },
            {
                "Sid": "InvokeSam3Endpoint",
                "Effect": "Allow",
                "Action": "sagemaker:InvokeEndpointAsync",
                "Resource": (
                    f"arn:aws:sagemaker:{region}:{account_id}:endpoint/{prefix}-sam3"
                ),
            },
            {
                "Sid": "PutInputPayload",
                "Effect": "Allow",
                "Action": ["s3:PutObject"],
                "Resource": f"{arns[0]}/*",
            },
            {
                "Sid": "ReadOutputResult",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [arns[1], f"{arns[1]}/*"],
            },
        ],
    })),
)


# ---------------------------------------------------------------------------
# Fine-tuning (SageMaker Training Jobs) — ECR repo + roles
# ---------------------------------------------------------------------------
# Training jobs are created on demand by Django (boto3 create_training_job), so
# there is no always-on resource here — just the image repo and the IAM the job
# + Django need. The training container (sagemaker_backend/Dockerfile.training)
# reads a manifest, assembles a YOLO dataset from annotated frames, trains, and
# writes best.pt to the models bucket + result.json to the output bucket.

training_ecr_repo = aws.ecr.Repository(
    "training-repo",
    name=f"{prefix}-training",  # must match the CI workflow
    image_tag_mutability="MUTABLE",
    image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(
        scan_on_push=True,
    ),
    force_delete=False,
)

aws.ecr.LifecyclePolicy(
    "training-repo-lifecycle",
    repository=training_ecr_repo.name,
    policy=json.dumps({
        "rules": [{
            "rulePriority": 1,
            "description": "Expire untagged images after 14 days",
            "selection": {
                "tagStatus": "untagged",
                "countType": "sinceImagePushed",
                "countUnit": "days",
                "countNumber": 14,
            },
            "action": {"type": "expire"},
        }],
    }),
)

# Execution role assumed by the training container.
#   - Read the manifest (sm-input) + raw videos.
#   - Write best.pt to the models bucket + result.json to sm-output.
#   - Pull ECR, write CloudWatch.
training_role = aws.iam.Role(
    "sagemaker-training-role",
    name=f"{prefix}-training-exec-role",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }),
)

aws.iam.RolePolicy(
    "sagemaker-training-policy",
    role=training_role.id,
    policy=pulumi.Output.all(input_bucket.arn, output_bucket.arn).apply(
        lambda arns: json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "ReadManifest",
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:ListBucket"],
                    "Resource": [arns[0], f"{arns[0]}/*"],
                },
                {
                    "Sid": "WriteResult",
                    "Effect": "Allow",
                    "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
                    "Resource": [arns[1], f"{arns[1]}/*"],
                },
                {
                    "Sid": "ReadRawVideos",
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:ListBucket"],
                    "Resource": [web_raw_videos_arn, f"{web_raw_videos_arn}/*"],
                },
                {
                    "Sid": "WriteModels",
                    "Effect": "Allow",
                    "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
                    "Resource": [web_models_arn, f"{web_models_arn}/*"],
                },
                {
                    "Sid": "ECRPull",
                    "Effect": "Allow",
                    "Action": [
                        "ecr:GetAuthorizationToken",
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage",
                    ],
                    "Resource": "*",
                },
                {
                    "Sid": "CloudWatch",
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "logs:DescribeLogStreams",
                        "cloudwatch:PutMetricData",
                    ],
                    "Resource": "*",
                },
            ],
        })
    ),
)

# Managed policy for the Django (App Runner) role: create/describe/stop training
# jobs, pass the training execution role to SageMaker, and write the manifest /
# read the result. Attached by ARN in the web stack (no StackReference).
django_training_policy = aws.iam.Policy(
    "django-training-policy",
    name=f"{prefix}-django-training",
    description="Django web app: create/describe BeeMonitor SageMaker training jobs + S3 I/O",
    policy=pulumi.Output.all(
        input_bucket.arn, output_bucket.arn, training_role.arn,
    ).apply(lambda a: json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ManageTrainingJobs",
                "Effect": "Allow",
                "Action": [
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:StopTrainingJob",
                    "sagemaker:AddTags",
                ],
                "Resource": f"arn:aws:sagemaker:{region}:{account_id}:training-job/*",
            },
            {
                "Sid": "PassTrainingRole",
                "Effect": "Allow",
                "Action": "iam:PassRole",
                "Resource": a[2],
                "Condition": {"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}},
            },
            {
                "Sid": "PutManifest",
                "Effect": "Allow",
                "Action": ["s3:PutObject"],
                "Resource": f"{a[0]}/*",
            },
            {
                "Sid": "ReadTrainingResult",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [a[1], f"{a[1]}/*"],
            },
        ],
    })),
)


# ---------------------------------------------------------------------------
# SageMaker Model / EndpointConfig / Endpoint / Autoscaling
# ---------------------------------------------------------------------------
# Gated on deploy-endpoint so pass 1 (no image yet) doesn't try to pull a
# non-existent image and stall the stack.

if deploy_endpoint:
    image_uri = pulumi.Output.concat(
        ecr_repo.repository_url, ":", image_tag,
    )

    # Suffix Model + EndpointConfig names with the image tag so a new tag yields
    # new resource names -> the Endpoint updates to the new config (a real
    # rolling deploy). With fixed names, an image change only replaces the Model
    # and the live endpoint keeps serving the old image.
    tag_slug = "".join(c for c in image_tag if c.isalnum())[:12] or "latest"
    # Bump when the EndpointConfig SHAPE changes without an image bump (the
    # name must change for the replace-then-update roll to work — same-name
    # re-creation collides with the retained old config).
    CONFIG_REV = "r5"
    tag_slug = f"{tag_slug}-{CONFIG_REV}"

    model = aws.sagemaker.Model(
        "model",
        name=f"{prefix}-model-{tag_slug}",
        execution_role_arn=sagemaker_role.arn,
        primary_container=aws.sagemaker.ModelPrimaryContainerArgs(
            image=image_uri,
            mode="SingleModel",
            # Env vars the inference container reads at boot.
            environment={
                "AWS_S3_BUCKET_RAW_VIDEOS": f"beemonitor-{env}-raw-videos-{account_id}",
                "AWS_S3_BUCKET_PROCESSED": f"beemonitor-{env}-processed-{account_id}",
                "AWS_S3_BUCKET_MODELS": f"beemonitor-{env}-models-{account_id}",
                "AWS_S3_BUCKET_USER_CONFIGS": f"beemonitor-{env}-user-configs-{account_id}",
                "AWS_REGION": region,
            },
        ),
        # Retain on replace so an image bump doesn't delete the old model mid-roll
        # (see the sam3_model note — same UpdateEndpoint race).
        opts=pulumi.ResourceOptions(retain_on_delete=True),
    )

    endpoint_config = aws.sagemaker.EndpointConfiguration(
        "endpoint-config",
        name=f"{prefix}-config-{tag_slug}",
        production_variants=[
            aws.sagemaker.EndpointConfigurationProductionVariantArgs(
                variant_name=VARIANT_NAME,
                model_name=model.name,
                instance_type=instance_type,
                initial_instance_count=1,
            ),
        ],
        async_inference_config=aws.sagemaker.EndpointConfigurationAsyncInferenceConfigArgs(
            # Videos per instance. One YOLO tracking job barely uses the T4 (the
            # work is mostly CPU/IO), so pack several per instance to clear big
            # batches faster. Safe now that the annotated-video memory leak is
            # fixed (was the 2026-07-06 OOM) and each invocation is isolated
            # (own temp dir + own YOLO instance). Kept <= the container's
            # gunicorn --threads 4 so a thread stays free for /ping health.
            client_config=aws.sagemaker.EndpointConfigurationAsyncInferenceConfigClientConfigArgs(
                max_concurrent_invocations_per_instance=3,
            ),
            output_config=aws.sagemaker.EndpointConfigurationAsyncInferenceConfigOutputConfigArgs(
                s3_output_path=pulumi.Output.concat("s3://", output_bucket.bucket, "/"),
                # Without a failure path, platform-level failures (container
                # crash/OOM, 6h request-TTL expiry) are dropped silently and
                # jobs look "Processing" forever. The web poller reads the
                # FailureLocation returned by invoke_endpoint_async.
                s3_failure_path=pulumi.Output.concat("s3://", output_bucket.bucket, "/failures/"),
            ),
        ),
        opts=pulumi.ResourceOptions(retain_on_delete=True),  # see model note
    )

    endpoint = aws.sagemaker.Endpoint(
        "endpoint",
        name=prefix,
        endpoint_config_name=endpoint_config.name,
    )

    # Application Auto Scaling — scale to zero after idle.
    # depends_on the endpoint (not just its name Output): an image bump REPLACES the
    # EndpointConfig and updates the endpoint, which takes minutes. Without explicit
    # ordering, Pulumi runs RegisterScalableTarget concurrently with that swap, and it
    # fails ("Could not find endpointConfig …") because the config is in flux — which
    # aborts the run mid-update and leaves the endpoint pointing at a deleted config.
    # Waiting for the endpoint update to settle first makes every roll clean.
    autoscaling_target = aws.appautoscaling.Target(
        "autoscaling-target",
        max_capacity=max_capacity,
        min_capacity=0,
        resource_id=pulumi.Output.concat(
            "endpoint/", endpoint.name, "/variant/", VARIANT_NAME,
        ),
        scalable_dimension="sagemaker:variant:DesiredInstanceCount",
        service_namespace="sagemaker",
        opts=pulumi.ResourceOptions(depends_on=[endpoint]),
    )

    aws.appautoscaling.Policy(
        "autoscaling-policy",
        name=f"{prefix}-scaling-policy",
        policy_type="TargetTrackingScaling",
        resource_id=autoscaling_target.resource_id,
        scalable_dimension=autoscaling_target.scalable_dimension,
        service_namespace=autoscaling_target.service_namespace,
        target_tracking_scaling_policy_configuration=aws.appautoscaling
            .PolicyTargetTrackingScalingPolicyConfigurationArgs(
                target_value=5.0,  # avg in-flight requests per instance
                customized_metric_specification=aws.appautoscaling
                    .PolicyTargetTrackingScalingPolicyConfigurationCustomizedMetricSpecificationArgs(
                        metric_name="ApproximateBacklogSizePerInstance",
                        namespace="AWS/SageMaker",
                        statistic="Average",
                    ),
                scale_in_cooldown=180,   # wait 3 min idle before scaling in (was 600)
                scale_out_cooldown=60,
            ),
    )

    # Scale-FROM-zero. The target-tracking policy above can scale toward min=0,
    # but an async endpoint only actually reaches 0 when there's a registered way
    # back UP — otherwise AWS pins it at the initial instance count forever (this
    # is exactly why beemonitor-sm-dev sat at 1 g4dn.xlarge 24/7 ≈ $530/mo with
    # zero traffic, while the otherwise-identical ecomorph endpoint cycles 1↔0).
    # A step policy adds 1 instance the moment a request is queued with no
    # capacity, driven by the HasBacklogWithoutCapacity alarm below. Mirrors the
    # working ecomorph endpoint exactly.
    scale_from_zero = aws.appautoscaling.Policy(
        "scale-from-zero",
        name=f"{prefix}-scale-from-zero",
        policy_type="StepScaling",
        resource_id=autoscaling_target.resource_id,
        scalable_dimension=autoscaling_target.scalable_dimension,
        service_namespace=autoscaling_target.service_namespace,
        step_scaling_policy_configuration=aws.appautoscaling
            .PolicyStepScalingPolicyConfigurationArgs(
                adjustment_type="ChangeInCapacity",
                cooldown=300,
                metric_aggregation_type="Maximum",
                step_adjustments=[
                    aws.appautoscaling
                    .PolicyStepScalingPolicyConfigurationStepAdjustmentArgs(
                        metric_interval_lower_bound="0",
                        scaling_adjustment=1,
                    ),
                ],
            ),
    )

    # Fires when the endpoint has a queued request but 0 instances to serve it.
    # TreatMissingData="missing" is essential: when idle the metric reports no
    # data, which must NOT be read as "has backlog" (that would wake it forever).
    aws.cloudwatch.MetricAlarm(
        "has-backlog-without-capacity",
        name=f"{prefix}-has-backlog-without-capacity",
        namespace="AWS/SageMaker",
        metric_name="HasBacklogWithoutCapacity",
        dimensions={"EndpointName": endpoint.name},
        statistic="Maximum",
        period=60,
        evaluation_periods=1,
        threshold=1,
        comparison_operator="GreaterThanOrEqualToThreshold",
        treat_missing_data="missing",
        alarm_actions=[scale_from_zero.arn],
    )

    pulumi.export("endpoint_name", endpoint.name)
    pulumi.export("endpoint_arn", endpoint.arn)


# ---------------------------------------------------------------------------
# BioCLIP insect-ID endpoint — SageMaker Serverless (CPU, scale-to-zero)
# ---------------------------------------------------------------------------
# Separate from the video endpoint: each request is one tiny crop, so Serverless
# Inference fits (sync, small payload) and scales to zero natively — no instance
# to keep alive, no autoscaling plumbing, no idle bill (the failure mode we just
# fixed on the GPU endpoint can't happen here). Gated on deploy-bioclip so the
# ECR image is built first (same two-pass flow as the video endpoint).
if deploy_bioclip:
    bioclip_tag = config.get("bioclip-image-tag") or "latest"
    bioclip_image = pulumi.Output.concat(
        bioclip_ecr_repo.repository_url, ":", bioclip_tag)
    bioclip_tag_slug = "".join(c for c in bioclip_tag if c.isalnum())[:12] or "latest"

    bioclip_model = aws.sagemaker.Model(
        "bioclip-model",
        name=f"{prefix}-bioclip-model-{bioclip_tag_slug}",
        execution_role_arn=sagemaker_role.arn,
        primary_container=aws.sagemaker.ModelPrimaryContainerArgs(
            image=bioclip_image,
            mode="SingleModel",
            environment={"BIOCLIP_TOPK": str(config.get_int("bioclip-topk") or 5)},
        ),
    )

    bioclip_config = aws.sagemaker.EndpointConfiguration(
        "bioclip-endpoint-config",
        name=f"{prefix}-bioclip-config-{bioclip_tag_slug}",
        production_variants=[
            aws.sagemaker.EndpointConfigurationProductionVariantArgs(
                variant_name=VARIANT_NAME,
                model_name=bioclip_model.name,
                serverless_config=aws.sagemaker
                    .EndpointConfigurationProductionVariantServerlessConfigArgs(
                        # BioCLIP (CLIP ViT-B + torch) resident ~1.5-2 GB; give
                        # headroom (max serverless is 6144) for fast cold loads.
                        memory_size_in_mb=config.get_int("bioclip-memory") or 6144,
                        max_concurrency=config.get_int("bioclip-max-concurrency") or 5,
                    ),
            ),
        ],
    )

    bioclip_endpoint = aws.sagemaker.Endpoint(
        "bioclip-endpoint",
        name=f"{prefix}-bioclip",
        endpoint_config_name=bioclip_config.name,
    )

    pulumi.export("bioclip_endpoint_name", bioclip_endpoint.name)


# ---------------------------------------------------------------------------
# SAM 3 auto-labeler endpoint — GPU async, scale-to-zero
# ---------------------------------------------------------------------------
# GPU (SAM 3 needs a GPU + VRAM), so this mirrors the video endpoint's async +
# scale-to-zero plumbing rather than BioCLIP's serverless (CPU-only). Runs only at
# label time, so it sits at 0 instances until a labeling request queues it up.
# Gated on deploy-sam3 (two-pass: build image first, then flip the flag).
# UNIFIED IMAGE (Option A, 2026-07-14): the SAM 3 endpoint now runs the SAME
# image as the main video endpoint (ecr_repo:image_tag) on a g5/A10G, so it
# serves BOTH SAM 3 pre-annotation AND SAM 3 tracking. This retires the separate
# auto-labeler image and removes the two-image transformers/Sam3Model drift.
# The bespoke sam3_ecr_repo + sam3-image-tag are no longer used for the endpoint.
if deploy_sam3:
    sam3_image = pulumi.Output.concat(ecr_repo.repository_url, ":", image_tag)
    sam3_tag_slug = "".join(c for c in image_tag if c.isalnum())[:12] or "latest"
    sam3_tag_slug = f"{sam3_tag_slug}-u1"  # bump when this config SHAPE changes

    sam3_model = aws.sagemaker.Model(
        "sam3-model",
        name=f"{prefix}-sam3-model-{sam3_tag_slug}",
        execution_role_arn=sagemaker_role.arn,
        primary_container=aws.sagemaker.ModelPrimaryContainerArgs(
            image=sam3_image,
            mode="SingleModel",
            # Same full env as the main model — the unified image runs the whole
            # tracking pipeline here too (ensure_models pulls from the models
            # bucket, etc.), not just pre_annotate.
            environment={
                "AWS_REGION": region,
                "AWS_S3_BUCKET_RAW_VIDEOS": f"beemonitor-{env}-raw-videos-{account_id}",
                "AWS_S3_BUCKET_PROCESSED": f"beemonitor-{env}-processed-{account_id}",
                "AWS_S3_BUCKET_MODELS": f"beemonitor-{env}-models-{account_id}",
                "AWS_S3_BUCKET_USER_CONFIGS": f"beemonitor-{env}-user-configs-{account_id}",
            },
        ),
        # retain_on_delete: an image bump replaces the model+config. Deleting the OLD
        # config while SageMaker's async UpdateEndpoint is still switching to the new
        # one fails ("Could not find endpoint configuration …") and leaves the endpoint
        # dangling. Retaining old model/config removes that race — the roll is clean;
        # orphaned model/config objects are free and can be pruned occasionally.
        opts=pulumi.ResourceOptions(retain_on_delete=True),
    )

    sam3_config = aws.sagemaker.EndpointConfiguration(
        "sam3-endpoint-config",
        name=f"{prefix}-sam3-config-{sam3_tag_slug}",
        production_variants=[
            aws.sagemaker.EndpointConfigurationProductionVariantArgs(
                variant_name=VARIANT_NAME,
                model_name=sam3_model.name,
                instance_type=sam3_instance_type,
                initial_instance_count=1,
            ),
        ],
        async_inference_config=aws.sagemaker.EndpointConfigurationAsyncInferenceConfigArgs(
            output_config=aws.sagemaker.EndpointConfigurationAsyncInferenceConfigOutputConfigArgs(
                s3_output_path=pulumi.Output.concat("s3://", output_bucket.bucket, "/"),
                # Failure records for SAM 3 TRACKING jobs (the poller reads these
                # to surface a chunk/job failure instead of hanging).
                s3_failure_path=pulumi.Output.concat("s3://", output_bucket.bucket, "/failures/"),
            ),
        ),
        opts=pulumi.ResourceOptions(retain_on_delete=True),  # see sam3_model note
    )

    sam3_endpoint = aws.sagemaker.Endpoint(
        "sam3-endpoint",
        name=f"{prefix}-sam3",
        endpoint_config_name=sam3_config.name,
    )

    # Scale-to-zero: identical pattern to the video endpoint (target-tracking toward
    # min=0 + a step policy driven by the HasBacklogWithoutCapacity alarm to wake
    # from zero). depends_on the endpoint so an image bump rolls cleanly (see the
    # video endpoint's autoscaling_target comment).
    sam3_target = aws.appautoscaling.Target(
        "sam3-autoscaling-target",
        max_capacity=sam3_max_capacity,
        min_capacity=0,
        resource_id=pulumi.Output.concat(
            "endpoint/", sam3_endpoint.name, "/variant/", VARIANT_NAME,
        ),
        scalable_dimension="sagemaker:variant:DesiredInstanceCount",
        service_namespace="sagemaker",
        opts=pulumi.ResourceOptions(depends_on=[sam3_endpoint]),
    )

    aws.appautoscaling.Policy(
        "sam3-autoscaling-policy",
        name=f"{prefix}-sam3-scaling-policy",
        policy_type="TargetTrackingScaling",
        resource_id=sam3_target.resource_id,
        scalable_dimension=sam3_target.scalable_dimension,
        service_namespace=sam3_target.service_namespace,
        target_tracking_scaling_policy_configuration=aws.appautoscaling
            .PolicyTargetTrackingScalingPolicyConfigurationArgs(
                target_value=5.0,
                customized_metric_specification=aws.appautoscaling
                    .PolicyTargetTrackingScalingPolicyConfigurationCustomizedMetricSpecificationArgs(
                        metric_name="ApproximateBacklogSizePerInstance",
                        namespace="AWS/SageMaker",
                        statistic="Average",
                    ),
                scale_in_cooldown=180,   # 3 min idle before scale-in (was 600)
                scale_out_cooldown=60,
            ),
    )

    sam3_scale_from_zero = aws.appautoscaling.Policy(
        "sam3-scale-from-zero",
        name=f"{prefix}-sam3-scale-from-zero",
        policy_type="StepScaling",
        resource_id=sam3_target.resource_id,
        scalable_dimension=sam3_target.scalable_dimension,
        service_namespace=sam3_target.service_namespace,
        step_scaling_policy_configuration=aws.appautoscaling
            .PolicyStepScalingPolicyConfigurationArgs(
                adjustment_type="ChangeInCapacity",
                cooldown=300,
                metric_aggregation_type="Maximum",
                step_adjustments=[
                    aws.appautoscaling
                    .PolicyStepScalingPolicyConfigurationStepAdjustmentArgs(
                        metric_interval_lower_bound="0",
                        scaling_adjustment=1,
                    ),
                ],
            ),
    )

    aws.cloudwatch.MetricAlarm(
        "sam3-has-backlog-without-capacity",
        name=f"{prefix}-sam3-has-backlog-without-capacity",
        namespace="AWS/SageMaker",
        metric_name="HasBacklogWithoutCapacity",
        dimensions={"EndpointName": sam3_endpoint.name},
        statistic="Maximum",
        period=60,
        evaluation_periods=1,
        threshold=1,
        comparison_operator="GreaterThanOrEqualToThreshold",
        treat_missing_data="missing",
        alarm_actions=[sam3_scale_from_zero.arn],
    )

    pulumi.export("sam3_endpoint_name", sam3_endpoint.name)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
pulumi.export("aws_account_id", account_id)
pulumi.export("aws_region", region)
pulumi.export("ecr_repo_url", ecr_repo.repository_url)
pulumi.export("ecr_repo_arn", ecr_repo.arn)
pulumi.export("sm_input_bucket", input_bucket.bucket)
pulumi.export("sm_output_bucket", output_bucket.bucket)
pulumi.export("sagemaker_exec_role_arn", sagemaker_role.arn)
pulumi.export("django_invoke_policy_arn", django_invoke_policy.arn)
pulumi.export("sam3_ecr_repo_url", sam3_ecr_repo.repository_url)
pulumi.export("training_ecr_repo_url", training_ecr_repo.repository_url)
pulumi.export("sagemaker_training_role_arn", training_role.arn)
pulumi.export("django_training_policy_arn", django_training_policy.arn)
