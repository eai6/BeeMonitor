"""BeeMonitor Cloud Platform — Pulumi Infrastructure as Code.

Provisions on Azure:
  1. Resource Group
  2. Storage Account + Blob Containers (raw-videos, processed, models, user-configs)
  3. PostgreSQL Flexible Server + Database
  4. Redis Cache
  5. App Service Plan + Web App (Django)
  6. App Service (Celery worker via always-on WebJob or Container)
  7. Container Registry (for Docker images)
"""

import pulumi
import pulumi_azure_native as azure
from pulumi_azure_native import resources, storage, dbforpostgresql, redis, web, containerregistry
import pulumi_docker as docker

config = pulumi.Config()
env = config.get("environment") or "dev"
location = azure.config.location or "eastus"
app_sku = config.get("appServiceSku") or "B1"
pg_sku = config.get("postgresSkuName") or "Standard_B1ms"
pg_tier = config.get("postgresTier") or "Burstable"

prefix = f"beemonitor-{env}"

# ── 1. Resource Group ─────────────────────────────────────────────────

rg = resources.ResourceGroup(
    f"{prefix}-rg",
    resource_group_name=f"{prefix}-rg",
    location=location,
)

# ── 2. Storage Account + Containers ───────────────────────────────────

storage_account = storage.StorageAccount(
    f"{prefix}-storage",
    account_name=f"beemonitor{env}store",
    resource_group_name=rg.name,
    location=rg.location,
    sku=storage.SkuArgs(name=storage.SkuName.STANDARD_LRS),
    kind=storage.Kind.STORAGE_V2,
    access_tier=storage.AccessTier.HOT,
)

containers = ["raw-videos", "processed", "models", "user-configs"]
for name in containers:
    storage.BlobContainer(
        f"{prefix}-container-{name}",
        container_name=name,
        account_name=storage_account.name,
        resource_group_name=rg.name,
        public_access=storage.PublicAccess.NONE,
    )

# Get storage connection string
storage_keys = pulumi.Output.all(rg.name, storage_account.name).apply(
    lambda args: storage.list_storage_account_keys(
        resource_group_name=args[0],
        account_name=args[1],
    )
)
storage_connection_string = pulumi.Output.all(
    storage_account.name, storage_keys
).apply(
    lambda args: f"DefaultEndpointsProtocol=https;AccountName={args[0]};"
    f"AccountKey={args[1].keys[0].value};"
    f"EndpointSuffix=core.windows.net"
)

# ── 3. PostgreSQL Flexible Server ─────────────────────────────────────

pg_password = config.require_secret("postgresPassword")

pg_server = dbforpostgresql.Server(
    f"{prefix}-pg",
    server_name=f"{prefix}-pg",
    resource_group_name=rg.name,
    location=rg.location,
    version=dbforpostgresql.PostgresMajorVersion.POSTGRES_MAJOR_VERSION_16,
    sku=dbforpostgresql.SkuArgs(
        name=pg_sku,
        tier=getattr(dbforpostgresql.SkuTier, pg_tier.upper(), dbforpostgresql.SkuTier.BURSTABLE),
    ),
    storage=dbforpostgresql.StorageArgs(
        storage_size_gb=32,
    ),
    administrator_login="beemonitor",
    administrator_login_password=pg_password,
)

pg_database = dbforpostgresql.Database(
    f"{prefix}-db",
    database_name="beemonitor",
    server_name=pg_server.name,
    resource_group_name=rg.name,
)

# Allow Azure services to access PostgreSQL
dbforpostgresql.FirewallRule(
    f"{prefix}-pg-allow-azure",
    firewall_rule_name="AllowAzureServices",
    server_name=pg_server.name,
    resource_group_name=rg.name,
    start_ip_address="0.0.0.0",
    end_ip_address="0.0.0.0",
)

# ── 4. Redis Cache ────────────────────────────────────────────────────

redis_cache = redis.Redis(
    f"{prefix}-redis",
    name=f"{prefix}-redis",
    resource_group_name=rg.name,
    location=rg.location,
    sku=redis.SkuArgs(
        name=redis.SkuName.BASIC,
        family=redis.SkuFamily.C,
        capacity=0,  # 250MB, cheapest tier
    ),
    enable_non_ssl_port=False,
    minimum_tls_version="1.2",
)

redis_keys = pulumi.Output.all(rg.name, redis_cache.name).apply(
    lambda args: redis.list_redis_keys(
        resource_group_name=args[0],
        name=args[1],
    )
)

redis_connection_string = pulumi.Output.all(redis_cache.host_name, redis_keys).apply(
    lambda args: f"rediss://:{args[1].primary_key}@{args[0]}:6380/0"
)

# ── 5. Container Registry ────────────────────────────────────────────

acr = containerregistry.Registry(
    f"{prefix}-acr",
    registry_name=f"beemonitor{env}acr",
    resource_group_name=rg.name,
    location=rg.location,
    sku=containerregistry.SkuArgs(name="Basic"),
    admin_user_enabled=True,
)

acr_credentials = pulumi.Output.all(rg.name, acr.name).apply(
    lambda args: containerregistry.list_registry_credentials(
        resource_group_name=args[0],
        registry_name=args[1],
    )
)

# ── 6. App Service Plan ──────────────────────────────────────────────

app_plan = web.AppServicePlan(
    f"{prefix}-plan",
    name=f"{prefix}-plan",
    resource_group_name=rg.name,
    location=rg.location,
    kind="Linux",
    reserved=True,
    sku=web.SkuDescriptionArgs(
        name=app_sku,
        tier="Basic" if app_sku.startswith("B") else "PremiumV3",
    ),
)

# ── 7. Web App (Django) ──────────────────────────────────────────────

django_app = web.WebApp(
    f"{prefix}-web",
    name=f"{prefix}-web",
    resource_group_name=rg.name,
    location=rg.location,
    server_farm_id=app_plan.id,
    site_config=web.SiteConfigArgs(
        linux_fx_version=pulumi.Output.all(acr.login_server).apply(
            lambda args: f"DOCKER|{args[0]}/beemonitor-web:latest"
        ),
        always_on=True,
        app_settings=[
            web.NameValuePairArgs(name="DJANGO_ENV", value="production"),
            web.NameValuePairArgs(name="DJANGO_SECRET_KEY", value=config.require_secret("djangoSecretKey")),
            web.NameValuePairArgs(name="AZURE_STORAGE_CONNECTION_STRING", value=storage_connection_string),
            web.NameValuePairArgs(name="CELERY_BROKER_URL", value=redis_connection_string),
            web.NameValuePairArgs(name="CELERY_RESULT_BACKEND", value=redis_connection_string),
            web.NameValuePairArgs(
                name="DB_HOST",
                value=pg_server.fully_qualified_domain_name,
            ),
            web.NameValuePairArgs(name="DB_NAME", value="beemonitor"),
            web.NameValuePairArgs(name="DB_USER", value="beemonitor"),
            web.NameValuePairArgs(name="DB_PASSWORD", value=pg_password),
            web.NameValuePairArgs(name="DB_PORT", value="5432"),
            web.NameValuePairArgs(
                name="ALLOWED_HOSTS",
                value=pulumi.Output.concat(f"{prefix}-web", ".azurewebsites.net"),
            ),
            web.NameValuePairArgs(
                name="BEEMONITOR_CREDENTIAL_KEY",
                value=config.require_secret("credentialKey"),
            ),
            web.NameValuePairArgs(name="MODAL_TOKEN_ID", value=config.get_secret("modalTokenId") or ""),
            web.NameValuePairArgs(name="MODAL_TOKEN_SECRET", value=config.get_secret("modalTokenSecret") or ""),
            # Docker registry
            web.NameValuePairArgs(name="DOCKER_REGISTRY_SERVER_URL", value=pulumi.Output.concat("https://", acr.login_server)),
            web.NameValuePairArgs(
                name="DOCKER_REGISTRY_SERVER_USERNAME",
                value=acr_credentials.apply(lambda c: c.username),
            ),
            web.NameValuePairArgs(
                name="DOCKER_REGISTRY_SERVER_PASSWORD",
                value=acr_credentials.apply(lambda c: c.passwords[0].value),
            ),
            web.NameValuePairArgs(name="WEBSITES_PORT", value="8000"),
        ],
    ),
    https_only=True,
)

# ── Outputs ───────────────────────────────────────────────────────────

pulumi.export("resource_group", rg.name)
pulumi.export("storage_account", storage_account.name)
pulumi.export("storage_connection_string", storage_connection_string)
pulumi.export("postgres_host", pg_server.fully_qualified_domain_name)
pulumi.export("redis_host", redis_cache.host_name)
pulumi.export("acr_login_server", acr.login_server)
pulumi.export("web_app_url", django_app.default_host_name.apply(lambda h: f"https://{h}"))
pulumi.export("web_app_name", django_app.name)
