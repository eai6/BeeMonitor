"""PWA views: manifest, service worker, offline page."""

from django.http import JsonResponse
from django.views.generic import TemplateView


def manifest(request):
    """Serve the web app manifest as JSON."""
    data = {
        "name": "BeeMonitor",
        "short_name": "BeeMonitor",
        "description": "AI-powered bee monitoring and analysis",
        "start_url": "/",
        "display": "standalone",
        "orientation": "portrait",
        "theme_color": "#F59E0B",
        "background_color": "#FFFFFF",
        "icons": [
            {
                "src": "/static/icons/icon-192.png",
                "sizes": "192x192",
                "type": "image/png",
            },
            {
                "src": "/static/icons/icon-512.png",
                "sizes": "512x512",
                "type": "image/png",
            },
        ],
        "categories": ["science", "productivity"],
    }
    return JsonResponse(data, content_type="application/manifest+json")


def service_worker(request):
    """Serve service worker from root scope."""
    from django.http import HttpResponse
    from pathlib import Path

    sw_path = Path(__file__).resolve().parent.parent.parent / "static" / "js" / "sw.js"
    if sw_path.exists():
        content = sw_path.read_text()
    else:
        content = "// Service worker placeholder"
    return HttpResponse(content, content_type="application/javascript")


class OfflineView(TemplateView):
    template_name = "pwa/offline.html"
