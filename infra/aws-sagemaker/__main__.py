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
max_capacity = config.get_int("max-capacity") or 2

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
# SageMaker Model / EndpointConfig / Endpoint / Autoscaling
# ---------------------------------------------------------------------------
# Gated on deploy-endpoint so pass 1 (no image yet) doesn't try to pull a
# non-existent image and stall the stack.

if deploy_endpoint:
    image_uri = pulumi.Output.concat(
        ecr_repo.repository_url, ":", image_tag,
    )

    model = aws.sagemaker.Model(
        "model",
        name=f"{prefix}-model",
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
    )

    endpoint_config = aws.sagemaker.EndpointConfiguration(
        "endpoint-config",
        name=f"{prefix}-config",
        production_variants=[
            aws.sagemaker.EndpointConfigurationProductionVariantArgs(
                variant_name=VARIANT_NAME,
                model_name=model.name,
                instance_type=instance_type,
                initial_instance_count=1,
            ),
        ],
        async_inference_config=aws.sagemaker.EndpointConfigurationAsyncInferenceConfigArgs(
            output_config=aws.sagemaker.EndpointConfigurationAsyncInferenceConfigOutputConfigArgs(
                s3_output_path=pulumi.Output.concat("s3://", output_bucket.bucket, "/"),
            ),
        ),
    )

    endpoint = aws.sagemaker.Endpoint(
        "endpoint",
        name=prefix,
        endpoint_config_name=endpoint_config.name,
    )

    # Application Auto Scaling — scale to zero after idle.
    autoscaling_target = aws.appautoscaling.Target(
        "autoscaling-target",
        max_capacity=max_capacity,
        min_capacity=0,
        resource_id=pulumi.Output.concat(
            "endpoint/", endpoint.name, "/variant/", VARIANT_NAME,
        ),
        scalable_dimension="sagemaker:variant:DesiredInstanceCount",
        service_namespace="sagemaker",
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
                scale_in_cooldown=600,   # wait 10 min idle before scaling in
                scale_out_cooldown=60,
            ),
    )

    pulumi.export("endpoint_name", endpoint.name)
    pulumi.export("endpoint_arn", endpoint.arn)


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
