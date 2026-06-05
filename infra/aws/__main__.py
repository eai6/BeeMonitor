"""
Pulumi IaC for BeeMonitor on AWS — Phase 2a (foundation).

Lays the bedrock that later phases build on:
  - the four S3 buckets created out-of-band in Phase 1a (imported here so
    Pulumi becomes the source of truth for their config);
  - the ECR repository the Django image will be pushed to;
  - the GitHub Actions OIDC role CI uses to push that image and (later)
    deploy infra changes;
  - an ECS task role + matching policy the Django app will run as in
    Phase 2b;
  - empty Secrets Manager entries for the secrets Django needs at runtime
    (values are filled out-of-band via `aws secretsmanager put-secret-value`
    so they never appear in source or state).

Compute resources (RDS, Fargate, ALB, ACM, Route53) come in Phase 2b. See
`memory/09_aws_migration_plan.md`.
"""

import json

import pulumi
import pulumi_aws as aws
import pulumi_random as random


# ---------------------------------------------------------------------------
# Config + identity
# ---------------------------------------------------------------------------

config = pulumi.Config()
env = config.get("environment") or "dev"
github_repo = config.require("github-repo")  # e.g. "owner/repo"
image_tag = config.get("image-tag") or "latest"
deploy_service = config.get_bool("deploy-service") or False
instance_cpu = config.get("instance-cpu") or "1024"   # 1 vCPU
instance_memory = config.get("instance-memory") or "2048"  # 2 GB
# Auto-analysis toggle. When false, the app gets an empty SAGEMAKER_ENDPOINT_NAME
# so Pi uploads do NOT spawn analysis jobs (useful while testing the pipeline).
_analysis_enabled = config.get_bool("analysis-enabled")
analysis_enabled = True if _analysis_enabled is None else _analysis_enabled

prefix = f"beemonitor-{env}"
account_id = aws.get_caller_identity().account_id
region = aws.get_region().name


# ---------------------------------------------------------------------------
# S3 buckets (imported from Phase 1a)
# ---------------------------------------------------------------------------
# The bucket *names* are already created in AWS (Phase 1a used `aws s3api
# create-bucket`). We import them so Pulumi manages their config — versioning,
# encryption, public-access-block, and the raw-videos lifecycle. The
# `aws s3 import` step is part of stack-init; see infra/aws/README before the
# first `pulumi up`.

def _bucket(logical: str, area: str) -> aws.s3.BucketV2:
    name = f"{prefix}-{area}-{account_id}"
    bucket = aws.s3.BucketV2(
        logical,
        bucket=name,
        # `force_destroy=False` is the default; spelled out so we can flip it
        # if we ever genuinely want to tear a bucket down.
        force_destroy=False,
        opts=pulumi.ResourceOptions(
            # Imported resources keep their existing AWS-side configuration on
            # the first run. Protect to stop accidental `pulumi destroy`.
            protect=True,
        ),
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
    aws.s3.BucketVersioningV2(
        f"{logical}-versioning",
        bucket=bucket.id,
        versioning_configuration=aws.s3.BucketVersioningV2VersioningConfigurationArgs(
            status="Enabled",
        ),
    )
    return bucket


raw_videos_bucket = _bucket("raw-videos", "raw-videos")
processed_bucket = _bucket("processed", "processed")
models_bucket = _bucket("models", "models")
user_configs_bucket = _bucket("user-configs", "user-configs")

# Raw videos transition to cheaper tiers — these are large, infrequently
# accessed after initial analysis. Mirrors what we set with the CLI in 1a.
aws.s3.BucketLifecycleConfigurationV2(
    "raw-videos-lifecycle",
    bucket=raw_videos_bucket.id,
    rules=[aws.s3.BucketLifecycleConfigurationV2RuleArgs(
        id="raw-videos-tiering",
        status="Enabled",
        filter=aws.s3.BucketLifecycleConfigurationV2RuleFilterArgs(prefix=""),
        transitions=[
            aws.s3.BucketLifecycleConfigurationV2RuleTransitionArgs(
                days=30, storage_class="STANDARD_IA",
            ),
            aws.s3.BucketLifecycleConfigurationV2RuleTransitionArgs(
                days=90, storage_class="GLACIER",
            ),
        ],
        noncurrent_version_expiration=aws.s3
            .BucketLifecycleConfigurationV2RuleNoncurrentVersionExpirationArgs(
                noncurrent_days=30,
            ),
    )],
)


# ---------------------------------------------------------------------------
# ECR — Django container image
# ---------------------------------------------------------------------------

ecr_repo = aws.ecr.Repository(
    "web-repo",
    name=f"{prefix}-web",
    image_tag_mutability="MUTABLE",  # CI re-pushes the `latest` tag on main
    image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(
        scan_on_push=True,
    ),
    force_delete=False,
)

aws.ecr.LifecyclePolicy(
    "web-repo-lifecycle",
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
# IAM — GitHub Actions OIDC role
# ---------------------------------------------------------------------------
# The OIDC provider was already created for EcoMorph in this account; reference
# it by ARN instead of creating a duplicate (an account can only have one
# provider per issuer).

GH_PROVIDER = "token.actions.githubusercontent.com"
GH_PROVIDER_ARN = f"arn:aws:iam::{account_id}:oidc-provider/{GH_PROVIDER}"

# Trust scoped to main branch and version tags only — PRs and feature
# branches cannot assume this role.
gh_role = aws.iam.Role(
    "github-actions-role",
    name=f"{prefix}-github-actions",
    description="GitHub Actions OIDC role for BeeMonitor CI",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Federated": GH_PROVIDER_ARN},
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {f"{GH_PROVIDER}:aud": "sts.amazonaws.com"},
                "StringLike": {
                    f"{GH_PROVIDER}:sub": [
                        f"repo:{github_repo}:ref:refs/heads/main",
                        f"repo:{github_repo}:ref:refs/tags/*",
                    ],
                },
            },
        }],
    }),
)

# Least-privilege: push to BOTH ECR repos that CI builds for —
# `beemonitor-{env}-web` (this stack) and `beemonitor-sm-{env}` (the
# SageMaker stack). The SageMaker ECR repo's ARN follows the same
# naming convention so we can reference it directly rather than via
# StackReference.
sm_ecr_repo_arn = (
    f"arn:aws:ecr:{region}:{account_id}:repository/beemonitor-sm-{env}"
)
ecr_push_policy_doc = ecr_repo.arn.apply(lambda repo_arn: json.dumps({
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "EcrLogin",
            "Effect": "Allow",
            "Action": ["ecr:GetAuthorizationToken"],
            "Resource": "*",
        },
        {
            "Sid": "EcrPushPullBeeMonitorRepos",
            "Effect": "Allow",
            "Action": [
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload",
                "ecr:PutImage",
                "ecr:DescribeRepositories",
                "ecr:DescribeImages",
            ],
            "Resource": [repo_arn, sm_ecr_repo_arn],
        },
    ],
}))

aws.iam.RolePolicy(
    "github-actions-ecr-push",
    role=gh_role.id,
    name="ecr-push",
    policy=ecr_push_policy_doc,
)


# ---------------------------------------------------------------------------
# IAM — ECS task role (Django runtime)
# ---------------------------------------------------------------------------
# What the running container is allowed to do. Phase 2b attaches this role
# to the Fargate task definition. Phase 4 extends the policy with the
# `sagemaker:InvokeEndpointAsync` permission.

task_role = aws.iam.Role(
    "ecs-task-role",
    name=f"{prefix}-ecs-task",
    description="Runtime role for the BeeMonitor Django container",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }),
)

# S3: read/write the four BeeMonitor buckets. No wildcard across all buckets.
task_s3_policy_doc = pulumi.Output.all(
    raw_videos_bucket.arn, processed_bucket.arn,
    models_bucket.arn, user_configs_bucket.arn,
).apply(lambda arns: json.dumps({
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "S3ReadWriteBeeMonitorBuckets",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "s3:GetObjectAttributes",
            ],
            "Resource": [
                *arns,
                *(f"{a}/*" for a in arns),
            ],
        },
    ],
}))

aws.iam.RolePolicy(
    "ecs-task-s3",
    role=task_role.id,
    name="s3-access",
    policy=task_s3_policy_doc,
)


# ECS task execution role — what ECS itself uses to pull images and write
# CloudWatch logs. Separate from the task role.
exec_role = aws.iam.Role(
    "ecs-exec-role",
    name=f"{prefix}-ecs-exec",
    description="ECS execution role (image pull + log delivery)",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }),
)

aws.iam.RolePolicyAttachment(
    "ecs-exec-managed",
    role=exec_role.name,
    policy_arn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
)


# ---------------------------------------------------------------------------
# Secrets Manager — Django runtime secrets (values filled out-of-band)
# ---------------------------------------------------------------------------
# Pulumi creates the secret *containers*; the actual values are set with
#   `aws secretsmanager put-secret-value --secret-id <id> --secret-string ...`
# so plaintext never enters source or Pulumi state.

def _secret(logical: str, name: str, description: str) -> aws.secretsmanager.Secret:
    secret = aws.secretsmanager.Secret(
        logical,
        name=f"{prefix}/{name}",
        description=description,
        # Default: AWS-managed KMS key (aws/secretsmanager). Switch to a
        # customer-managed key in 2b if compliance asks for it.
        recovery_window_in_days=7,
    )
    return secret


django_secret_key = _secret(
    "django-secret-key", "django-secret-key",
    "Django SECRET_KEY for the BeeMonitor web app",
)
db_password_secret = _secret(
    "db-password", "db-password",
    "RDS master password (used by Django and migrations)",
)
api_key_pepper = _secret(
    "api-key-pepper", "api-key-pepper",
    "Pepper appended to user API keys before hashing (apps/api/authentication.py)",
)
credential_key_secret = _secret(
    "credential-key", "credential-key",
    "Fernet key encrypting external cloud-source credentials (apps/sources/)",
)

# Populate the secret values via pulumi_random so they never appear in source.
# pulumi_random keeps the generated value in the (passphrase-encrypted) state
# only; rotating means changing the resource's `keepers` to force regeneration.
django_secret_value = random.RandomPassword(
    "django-secret-key-value",
    length=64,
    special=True,
    override_special="!@#$%^&*()-_=+",
)
aws.secretsmanager.SecretVersion(
    "django-secret-key-v1",
    secret_id=django_secret_key.id,
    secret_string=django_secret_value.result,
)

db_password_value = random.RandomPassword(
    "db-password-value",
    length=32,
    special=False,  # avoid URL-encoding hassles in DATABASE_URL
)
aws.secretsmanager.SecretVersion(
    "db-password-v1",
    secret_id=db_password_secret.id,
    secret_string=db_password_value.result,
)

api_key_pepper_value = random.RandomPassword(
    "api-key-pepper-value",
    length=64,
    special=False,
)
aws.secretsmanager.SecretVersion(
    "api-key-pepper-v1",
    secret_id=api_key_pepper.id,
    secret_string=api_key_pepper_value.result,
)

# Fernet keys are 32 url-safe base64 bytes — generate via RandomBytes.
credential_key_value = random.RandomBytes(
    "credential-key-value",
    length=32,
)
aws.secretsmanager.SecretVersion(
    "credential-key-v1",
    secret_id=credential_key_secret.id,
    # base64.urlsafe_b64encode of 32 bytes is what Fernet expects.
    secret_string=credential_key_value.base64,
)

# The task role needs to read these secrets at runtime. Restrict to the
# four secret ARNs above.
task_secrets_policy_doc = pulumi.Output.all(
    django_secret_key.arn, db_password_secret.arn,
    api_key_pepper.arn, credential_key_secret.arn,
).apply(lambda arns: json.dumps({
    "Version": "2012-10-17",
    "Statement": [{
        "Sid": "ReadBeeMonitorSecrets",
        "Effect": "Allow",
        "Action": [
            "secretsmanager:GetSecretValue",
            "secretsmanager:DescribeSecret",
        ],
        "Resource": list(arns),
    }],
}))

aws.iam.RolePolicy(
    "ecs-task-secrets",
    role=task_role.id,
    name="secrets-read",
    policy=task_secrets_policy_doc,
)


# ---------------------------------------------------------------------------
# VPC + subnets — App Runner VPC connector lives here, RDS lives here too.
# ---------------------------------------------------------------------------
# Pattern mirrors EcoMorph's web stack: a dedicated /16 with two private /20
# subnets across the first two AZs. No IGW / NAT — App Runner reaches AWS
# services via VPC endpoints below; outside-internet egress is intentionally
# absent (the container does not need it, and removing NAT saves ~$30/mo).

azs = aws.get_availability_zones(state="available").names[:2]

vpc = aws.ec2.Vpc(
    "web-vpc",
    cidr_block="10.30.0.0/16",
    enable_dns_hostnames=True,
    enable_dns_support=True,
    tags={"Name": f"{prefix}-vpc"},
)

private_subnets = [
    aws.ec2.Subnet(
        f"private-subnet-{i}",
        vpc_id=vpc.id,
        cidr_block=f"10.30.{i * 16}.0/20",
        availability_zone=az,
        tags={"Name": f"{prefix}-private-{az}"},
    )
    for i, az in enumerate(azs)
]

rds_sg = aws.ec2.SecurityGroup(
    "rds-sg",
    vpc_id=vpc.id,
    description=f"{prefix} RDS PostgreSQL",
    tags={"Name": f"{prefix}-rds-sg"},
)

app_runner_sg = aws.ec2.SecurityGroup(
    "app-runner-sg",
    vpc_id=vpc.id,
    description=f"{prefix} App Runner VPC connector ENIs",
    egress=[aws.ec2.SecurityGroupEgressArgs(
        from_port=0, to_port=0, protocol="-1", cidr_blocks=["0.0.0.0/0"],
    )],
    tags={"Name": f"{prefix}-apprunner-sg"},
)

aws.ec2.SecurityGroupRule(
    "rds-from-apprunner",
    type="ingress",
    from_port=5432,
    to_port=5432,
    protocol="tcp",
    source_security_group_id=app_runner_sg.id,
    security_group_id=rds_sg.id,
)


# ---------------------------------------------------------------------------
# VPC endpoints — egress_type=VPC routes ALL container egress through the VPC.
# Without these the container can't talk to S3 or Secrets Manager and hangs.
# ---------------------------------------------------------------------------

# S3 gateway endpoint — free, attached to the VPC's default route table.
aws.ec2.VpcEndpoint(
    "s3-gateway-endpoint",
    vpc_id=vpc.id,
    service_name=f"com.amazonaws.{region}.s3",
    vpc_endpoint_type="Gateway",
    route_table_ids=[vpc.default_route_table_id],
)

# Interface endpoints need an SG that lets App Runner ENIs talk to them on 443.
vpc_endpoint_sg = aws.ec2.SecurityGroup(
    "vpc-endpoint-sg",
    vpc_id=vpc.id,
    description=f"{prefix} VPC interface endpoints (HTTPS from App Runner)",
    tags={"Name": f"{prefix}-vpc-endpoint-sg"},
)
aws.ec2.SecurityGroupRule(
    "vpc-endpoint-from-apprunner",
    type="ingress",
    from_port=443,
    to_port=443,
    protocol="tcp",
    source_security_group_id=app_runner_sg.id,
    security_group_id=vpc_endpoint_sg.id,
)

# Secrets Manager interface endpoint (~$7/mo) — container fetches Django
# SECRET_KEY + DB password at boot via the boto3 SecretsManager client.
aws.ec2.VpcEndpoint(
    "secretsmanager-endpoint",
    vpc_id=vpc.id,
    service_name=f"com.amazonaws.{region}.secretsmanager",
    vpc_endpoint_type="Interface",
    subnet_ids=[s.id for s in private_subnets],
    security_group_ids=[vpc_endpoint_sg.id],
    private_dns_enabled=True,
)

# SageMaker Runtime interface endpoint (~$7/mo) — the Django app calls
# invoke_endpoint_async (auto-analysis on Pi upload). With no NAT/IGW, App Runner
# otherwise can't reach runtime.sagemaker.<region>.amazonaws.com and every spawn
# times out after 60s (which also pinned DB connections during upload bursts).
aws.ec2.VpcEndpoint(
    "sagemaker-runtime-endpoint",
    vpc_id=vpc.id,
    service_name=f"com.amazonaws.{region}.sagemaker.runtime",
    vpc_endpoint_type="Interface",
    subnet_ids=[s.id for s in private_subnets],
    security_group_ids=[vpc_endpoint_sg.id],
    private_dns_enabled=True,
)


# ---------------------------------------------------------------------------
# RDS Postgres 16 — the Django database
# ---------------------------------------------------------------------------

db_subnet_group = aws.rds.SubnetGroup(
    "db-subnet-group",
    name=f"{prefix}-db-subnet",
    subnet_ids=[s.id for s in private_subnets],
)

db_instance = aws.rds.Instance(
    "db",
    identifier=f"{prefix}-db",
    engine="postgres",
    engine_version="16",
    instance_class="db.t4g.micro",
    allocated_storage=20,
    storage_encrypted=True,
    db_name="beemonitor",
    username="beemonitor",
    password=db_password_value.result,
    db_subnet_group_name=db_subnet_group.name,
    vpc_security_group_ids=[rds_sg.id],
    publicly_accessible=False,
    multi_az=False,
    backup_retention_period=7,
    skip_final_snapshot=True,        # dev — no need to retain on destroy
    apply_immediately=True,
    auto_minor_version_upgrade=True,
)


# ---------------------------------------------------------------------------
# App Runner — instance role (runtime) and access role (ECR pull)
# ---------------------------------------------------------------------------
# These are App-Runner-specific. The ECS roles above stay around as a
# placeholder in case we ever ship a heavier worker on Fargate.

apprunner_instance_role = aws.iam.Role(
    "apprunner-instance-role",
    name=f"{prefix}-apprunner-instance-role",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "tasks.apprunner.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }),
)

# Same S3 policy the ECS task role uses — read/write the four BeeMonitor buckets.
aws.iam.RolePolicy(
    "apprunner-s3-access",
    role=apprunner_instance_role.id,
    name="s3-access",
    policy=task_s3_policy_doc,
)

# Read the four Secrets Manager secrets at container startup.
aws.iam.RolePolicy(
    "apprunner-secrets-read",
    role=apprunner_instance_role.id,
    name="secrets-read",
    policy=task_secrets_policy_doc,
)

apprunner_access_role = aws.iam.Role(
    "apprunner-access-role",
    name=f"{prefix}-apprunner-access-role",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "build.apprunner.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }),
)
aws.iam.RolePolicyAttachment(
    "apprunner-ecr-pull",
    role=apprunner_access_role.name,
    policy_arn="arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess",
)

# Attach the SageMaker django-invoke policy so the running container can
# invoke the BeeMonitor inference endpoint. The policy is created by the
# separate `infra/aws-sagemaker/` stack; the ARN follows naming convention
# so we don't need a StackReference.
sm_django_invoke_policy_arn = (
    f"arn:aws:iam::{account_id}:policy/beemonitor-sm-{env}-django-invoke"
)
aws.iam.RolePolicyAttachment(
    "apprunner-sagemaker-invoke",
    role=apprunner_instance_role.name,
    policy_arn=sm_django_invoke_policy_arn,
)


# ---------------------------------------------------------------------------
# App Runner service — gated on `deploy-service` (two-pass deploy)
# ---------------------------------------------------------------------------
# Pass 1: deploy-service=false → everything above gets created (VPC, RDS,
# roles, secret values populated). Then push the image to ECR.
# Pass 2: flip to true → App Runner service launches against that image.

if deploy_service:
    vpc_connector = aws.apprunner.VpcConnector(
        "vpc-connector",
        vpc_connector_name=f"{prefix}-vpc-connector",
        subnets=[s.id for s in private_subnets],
        security_groups=[app_runner_sg.id],
    )

    image_uri = pulumi.Output.concat(ecr_repo.repository_url, ":", image_tag)

    env_vars = pulumi.Output.all(
        db_instance.address,
        db_instance.port,
        db_instance.db_name,
        db_instance.username,
        raw_videos_bucket.bucket,
        processed_bucket.bucket,
        models_bucket.bucket,
        user_configs_bucket.bucket,
    ).apply(lambda args: {
        "DJANGO_ENV": "production",
        # App Runner's internal health check probes with an unpredictable Host
        # header; the wildcard ALLOWED_HOSTS is safe because only the
        # App Runner edge can reach the container.
        "ALLOWED_HOSTS": "*",
        "CSRF_TRUSTED_ORIGINS": "https://*.awsapprunner.com",
        "AWS_REGION": region,
        "DB_HOST": args[0],
        "DB_PORT": str(args[1]),
        "DB_NAME": args[2],
        "DB_USER": args[3],
        "AWS_S3_BUCKET_RAW_VIDEOS": args[4],
        "AWS_S3_BUCKET_PROCESSED": args[5],
        "AWS_S3_BUCKET_MODELS": args[6],
        "AWS_S3_BUCKET_USER_CONFIGS": args[7],
        # SageMaker (Phase 4) — set by convention; the SM stack owns the names.
        # Blanked when analysis-enabled=false so uploads don't spawn jobs.
        "SAGEMAKER_ENDPOINT_NAME": f"beemonitor-sm-{env}" if analysis_enabled else "",
        "SAGEMAKER_INPUT_BUCKET": f"beemonitor-sm-{env}-input-{account_id}",
        "SAGEMAKER_OUTPUT_BUCKET": f"beemonitor-sm-{env}-output-{account_id}",
    })

    runtime_secrets = pulumi.Output.all(
        django_secret_key.arn,
        db_password_secret.arn,
        api_key_pepper.arn,
        credential_key_secret.arn,
    ).apply(lambda arns: {
        "DJANGO_SECRET_KEY": arns[0],
        "DB_PASSWORD": arns[1],
        "BEEMONITOR_API_KEY_PEPPER": arns[2],
        "BEEMONITOR_CREDENTIAL_KEY": arns[3],
    })

    web_service = aws.apprunner.Service(
        "web",
        service_name=f"{prefix}-web",
        source_configuration=aws.apprunner.ServiceSourceConfigurationArgs(
            authentication_configuration=aws.apprunner.ServiceSourceConfigurationAuthenticationConfigurationArgs(
                access_role_arn=apprunner_access_role.arn,
            ),
            auto_deployments_enabled=True,
            image_repository=aws.apprunner.ServiceSourceConfigurationImageRepositoryArgs(
                image_identifier=image_uri,
                image_repository_type="ECR",
                image_configuration=aws.apprunner.ServiceSourceConfigurationImageRepositoryImageConfigurationArgs(
                    port="8000",
                    runtime_environment_variables=env_vars,
                    runtime_environment_secrets=runtime_secrets,
                ),
            ),
        ),
        instance_configuration=aws.apprunner.ServiceInstanceConfigurationArgs(
            cpu=instance_cpu,
            memory=instance_memory,
            instance_role_arn=apprunner_instance_role.arn,
        ),
        network_configuration=aws.apprunner.ServiceNetworkConfigurationArgs(
            egress_configuration=aws.apprunner.ServiceNetworkConfigurationEgressConfigurationArgs(
                egress_type="VPC",
                vpc_connector_arn=vpc_connector.arn,
            ),
        ),
        health_check_configuration=aws.apprunner.ServiceHealthCheckConfigurationArgs(
            protocol="HTTP",
            # /accounts/login/ is the cheapest known-200 route on the Django app
            # (no DB writes, renders quickly). Same choice EcoMorph made.
            path="/accounts/login/",
            interval=10,
            timeout=5,
            healthy_threshold=1,
            unhealthy_threshold=5,
        ),
    )

    pulumi.export("web_service_url", pulumi.Output.concat("https://", web_service.service_url))


# ---------------------------------------------------------------------------
# Outputs — surfaced for CI workflows + the Phase 2b stack
# ---------------------------------------------------------------------------

pulumi.export("aws_account_id", account_id)
pulumi.export("aws_region", region)

pulumi.export("ecr_repo_url", ecr_repo.repository_url)
pulumi.export("ecr_repo_arn", ecr_repo.arn)

pulumi.export("github_actions_role_arn", gh_role.arn)
pulumi.export("ecs_task_role_arn", task_role.arn)
pulumi.export("ecs_exec_role_arn", exec_role.arn)

pulumi.export("secret_django_secret_key_arn", django_secret_key.arn)
pulumi.export("secret_db_password_arn", db_password_secret.arn)
pulumi.export("secret_api_key_pepper_arn", api_key_pepper.arn)
pulumi.export("secret_credential_key_arn", credential_key_secret.arn)

pulumi.export("bucket_raw_videos", raw_videos_bucket.bucket)
pulumi.export("bucket_processed", processed_bucket.bucket)
pulumi.export("bucket_models", models_bucket.bucket)
pulumi.export("bucket_user_configs", user_configs_bucket.bucket)

pulumi.export("vpc_id", vpc.id)
pulumi.export("private_subnet_ids", [s.id for s in private_subnets])
pulumi.export("rds_endpoint", db_instance.endpoint)
pulumi.export("rds_address", db_instance.address)
pulumi.export("apprunner_instance_role_arn", apprunner_instance_role.arn)
pulumi.export("apprunner_access_role_arn", apprunner_access_role.arn)
