"""Sending email, and remembering whether it went.

Every platform email goes through ``send`` so it lands in ``EmailLog`` with
SES's answer. A failure never raises to the caller: a password-reset page that
errors only for unknown addresses would tell an attacker which ones exist, and
a notice job should carry on with the next user. Callers that must know (the
footage-deletion notices) read the returned log row.

SES itself: while the account is in the SES sandbox it delivers only to
verified addresses and rejects everything else with ``MessageRejected``. That
is what broke password reset in October 2026; ``ses_status`` exposes it.
"""

from __future__ import annotations

import logging

from django.conf import settings
from django.core.mail import EmailMultiAlternatives

from .models import EmailLog

logger = logging.getLogger(__name__)


def send(kind, to, subject, body, html=None, user=None) -> EmailLog:
    """Send one email and log the outcome. Never raises."""
    msg = EmailMultiAlternatives(subject, body, settings.DEFAULT_FROM_EMAIL, [to])
    if html:
        msg.attach_alternative(html, "text/html")
    try:
        msg.send()
    except Exception as e:  # SES rejections, throttling, network, bad config
        logger.exception("email %s to %s failed", kind, to)
        return EmailLog.objects.create(
            to=to, user=user, kind=kind, subject=subject[:255],
            status=EmailLog.Status.FAILED, error=str(e)[:2000])
    # Anymail records SES's message id on the message; other backends don't.
    status = getattr(msg, "anymail_status", None)
    message_id = (getattr(status, "message_id", "") or "") if status else ""
    return EmailLog.objects.create(
        to=to, user=user, kind=kind, subject=subject[:255],
        status=EmailLog.Status.SENT, message_id=str(message_id)[:200])


def ses_status() -> dict:
    """What SES says about the account: sandboxed or not, today's quota.

    Returns ``{"ok": False, "error": ...}`` when it can't ask (no AWS access,
    the role lacks ses:GetAccount, a non-SES backend). Never raises.
    """
    if "amazon_ses" not in settings.EMAIL_BACKEND:
        return {"ok": False, "error": f"Email backend is {settings.EMAIL_BACKEND}, not SES."}
    try:
        import boto3
        acct = boto3.client("sesv2", region_name=settings.AWS_REGION).get_account()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    quota = acct.get("SendQuota") or {}
    return {
        "ok": True,
        "production": bool(acct.get("ProductionAccessEnabled")),
        "sending_enabled": bool(acct.get("SendingEnabled")),
        "sent_24h": int(quota.get("SentLast24Hours") or 0),
        "max_24h": int(quota.get("Max24HourSend") or 0),
        "review_case": ((acct.get("Details") or {}).get("ReviewDetails") or {}).get("CaseId", ""),
    }
