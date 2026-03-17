from rest_framework.permissions import BasePermission

from apps.accounts.models import UserProfile


class HasTierPermission(BasePermission):
    """
    Permission class that checks the user's subscription tier.

    Set `required_tier` on the view to specify the minimum tier required.
    Tier hierarchy: free < standard < pro.
    """

    TIER_HIERARCHY = {
        UserProfile.Tier.FREE: 0,
        UserProfile.Tier.STANDARD: 1,
        UserProfile.Tier.PRO: 2,
    }

    message = "Your subscription tier does not have access to this resource."

    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False

        required_tier = getattr(view, "required_tier", UserProfile.Tier.FREE)

        try:
            profile = request.user.profile
        except UserProfile.DoesNotExist:
            return False

        user_level = self.TIER_HIERARCHY.get(profile.tier, 0)
        required_level = self.TIER_HIERARCHY.get(required_tier, 0)

        return user_level >= required_level
