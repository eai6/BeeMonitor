from rest_framework.throttling import SimpleRateThrottle


class TierBasedThrottle(SimpleRateThrottle):
    """
    Throttle requests based on the user's subscription tier.

    Rates:
        - free:     10 requests per minute
        - standard: 60 requests per minute
        - pro:      300 requests per minute
    """

    TIER_RATES = {
        "free": "10/min",
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
                "free",
            )
            self.rate = self.TIER_RATES.get(tier, self.TIER_RATES["free"])
        else:
            self.rate = self.TIER_RATES["free"]

        self.num_requests, self.duration = self.parse_rate(self.rate)
        return super().allow_request(request, view)
