"""Ownership-scoping helpers for views.

Default behaviour everywhere in BeeMonitor: every read/write path filters by
``request.user`` so user B cannot see user A's resources. ``CrossUserScopingTests``
in ``tests.py`` locks that contract.

Support staff (``UserProfile.is_support = True``) are the only exception. They
need read-only cross-tenant visibility for analytics, debugging, billing
disputes. They can NOT mutate other users' data — writes still flow through
the ordinary owner-scoped views.

Usage::

    class JobListView(OwnedListViewMixin, LoginRequiredMixin, ListView):
        model = Job
        owner_field = "user"   # FK name on the model

The mixin returns ``request.user``'s own objects, or the full queryset for a
support reader. Writes (POST/PUT/PATCH/DELETE) are unaffected — they still
hit the normal ``get_queryset`` filter on the writing view.
"""

from __future__ import annotations


def is_support(user) -> bool:
    """True if this user has the support read-bypass flag set."""
    if not user.is_authenticated:
        return False
    try:
        return bool(user.profile.is_support)
    except Exception:
        return False


class OwnedListViewMixin:
    """Mixin for read-only list views.

    Filters by ``self.owner_field == request.user`` for normal users,
    returns the full queryset for support readers.
    """

    owner_field: str = "user"

    def get_queryset(self):
        qs = super().get_queryset()
        if is_support(self.request.user):
            return qs
        return qs.filter(**{self.owner_field: self.request.user})
