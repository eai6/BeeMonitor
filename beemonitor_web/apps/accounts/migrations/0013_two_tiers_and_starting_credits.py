"""Two tiers, and 25 credits once instead of 10 a month (memory/50 §2, §3).

Free becomes Standard and Enterprise becomes Pro (both were unlimited, so no
limits are lost). Standard accounts, which had a monthly allowance that no
longer exists, get the same 25 starting credits a new account gets, on top of
any granted credits they already have.
"""

from decimal import Decimal

from django.db import migrations

STARTING_CREDITS = Decimal(25)


def forwards(apps, schema_editor):
    UserProfile = apps.get_model("accounts", "UserProfile")
    Coupon = apps.get_model("accounts", "Coupon")
    UserProfile.objects.filter(tier="free").update(tier="standard")
    UserProfile.objects.filter(tier="enterprise").update(tier="pro")
    Coupon.objects.filter(upgrade_tier="enterprise").update(upgrade_tier="pro")
    Coupon.objects.filter(upgrade_tier="free").update(upgrade_tier="standard")
    for profile in UserProfile.objects.filter(tier="standard", user__is_superuser=False):
        profile.granted_gpu_credits += STARTING_CREDITS
        profile.save(update_fields=["granted_gpu_credits"])


class Migration(migrations.Migration):
    dependencies = [("accounts", "0012_standard_pro_and_requests")]

    operations = [migrations.RunPython(forwards, migrations.RunPython.noop)]
