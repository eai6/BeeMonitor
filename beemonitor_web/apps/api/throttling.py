from rest_framework.throttling import SimpleRateThrottle


class TierBasedThrottle(SimpleRateThrottle):
    """
    Throttle requests based on the user's subscription tier.

    Rates:
        - standard: 60 requests per minute
        - pro:      300 requests per minute
    """

    TIER_RATES = {
        "standard": "60/min",
        "pro": "300/min",
    }

    scope = "tier"

    def get_cache_key(self, request, view):
        if request.user and request.user.is_authenticated:
            return self.cache_format % {
                "scope": self.scope,
                "ident": request.user.pk,
            }
        return self.get_ident(request)

    def get_rate(self):
        return None

    def allow_request(self, request, view):
        if request.user and request.user.is_authenticated:
            tier = getattr(
                getattr(request.user, "profile", None),
                "tier",
                "standard",
            )
            self.rate = self.TIER_RATES.get(tier, self.TIER_RATES["standard"])
        else:
            self.rate = self.TIER_RATES["standard"]

        self.num_requests, self.duration = self.parse_rate(self.rate)
        return super().allow_request(request, view)
