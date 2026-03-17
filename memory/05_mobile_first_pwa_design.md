# BeeMonitor Mobile-First Offline PWA Design

## Overview

A Progressive Web Application (PWA) built from the Django web app that works offline-first, is mobile-optimized, and provides field researchers with essential BeeMonitor functionality on any device with a browser.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              MOBILE BROWSER (PWA)                │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │         Service Worker Layer                │  │
│  │  • Cache API (static + dynamic)            │  │
│  │  • Background Sync (queued uploads)        │  │
│  │  • Push Notifications (job complete)       │  │
│  └──────────────────┬─────────────────────────┘  │
│                     │                             │
│  ┌──────────────────┴─────────────────────────┐  │
│  │           Application Shell                 │  │
│  │  • HTMX + Alpine.js (reactive UI)          │  │
│  │  • Tailwind CSS (mobile-first)             │  │
│  │  • IndexedDB (offline data store)          │  │
│  │  • Camera API (direct capture)             │  │
│  └──────────────────┬─────────────────────────┘  │
│                     │                             │
│  ┌──────────────────┴─────────────────────────┐  │
│  │           Offline Data Layer                │  │
│  │  • IndexedDB: jobs, results, settings      │  │
│  │  • Cache Storage: static assets, API       │  │
│  │  • File API: queued video uploads          │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────┘
                       │ (when online)
                       ▼
┌──────────────────────────────────────────────────┐
│          DJANGO BACKEND (same as web app)         │
│  • REST API (DRF)                                │
│  • PWA manifest + service worker serving         │
│  • Push notification server (web-push)           │
└──────────────────────────────────────────────────┘
```

---

## PWA Configuration

### Web App Manifest
```json
{
  "name": "BeeMonitor",
  "short_name": "BeeMonitor",
  "description": "AI-powered bee monitoring and analysis",
  "start_url": "/dashboard/",
  "display": "standalone",
  "orientation": "portrait",
  "theme_color": "#F59E0B",
  "background_color": "#FFFFFF",
  "icons": [
    { "src": "/static/icons/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/static/icons/icon-512.png", "sizes": "512x512", "type": "image/png" },
    { "src": "/static/icons/icon-maskable.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
  ],
  "categories": ["science", "productivity"],
  "shortcuts": [
    { "name": "Upload Video", "url": "/videos/upload/", "icon": "/static/icons/upload.png" },
    { "name": "View Jobs", "url": "/analysis/", "icon": "/static/icons/jobs.png" }
  ]
}
```

---

## Offline Strategy

### Cache Layers

| Layer | What | Strategy | Storage |
|-------|------|----------|---------|
| App Shell | HTML, CSS, JS, icons | Cache-first | Cache API |
| API Responses | Job list, results, stats | Stale-while-revalidate | Cache API |
| User Data | Jobs, events, settings | IndexedDB-first | IndexedDB |
| Videos | Queued uploads | Store until sync | File API + IndexedDB |
| Images | Thumbnails, charts | Cache-first | Cache API |

### Service Worker Strategy

```javascript
// sw.js - Workbox-based service worker

// 1. App Shell: Cache on install, serve cache-first
precacheAndRoute([
  '/static/css/app.css',
  '/static/js/app.js',
  '/offline/',
  // ... all shell assets
]);

// 2. API calls: Network-first with cache fallback
registerRoute(
  /\/api\/v1\//,
  new NetworkFirst({
    cacheName: 'api-cache',
    networkTimeoutSeconds: 5,
    plugins: [new ExpirationPlugin({ maxEntries: 200, maxAgeSeconds: 86400 })]
  })
);

// 3. Page navigation: Network-first, fallback to offline page
registerRoute(
  ({ request }) => request.mode === 'navigate',
  new NetworkFirst({
    cacheName: 'pages',
    plugins: [new ExpirationPlugin({ maxEntries: 50 })]
  })
);

// 4. Background Sync: Queue uploads when offline
registerRoute(
  /\/api\/v1\/videos\/upload/,
  new NetworkOnly({
    plugins: [new BackgroundSyncPlugin('video-upload-queue', {
      maxRetentionTime: 7 * 24 * 60  // 7 days
    })]
  }),
  'POST'
);
```

### IndexedDB Schema

```javascript
// Dexie.js for IndexedDB management
const db = new Dexie('BeeMonitorOffline');
db.version(1).stores({
  jobs:     '++id, serverId, status, videoId, createdAt',
  results:  '++id, jobId, eventCount, entryCount, exitCount',
  events:   '++id, jobId, nestId, action, timestamp',
  videos:   '++id, serverId, title, status, localPath',
  settings: 'key',
  syncQueue:'++id, action, endpoint, payload, createdAt'
});
```

---

## Mobile-First UI Design

### Navigation (Bottom Tab Bar)
```
┌──────────────────────────────────────┐
│                                      │
│         [Page Content Area]          │
│                                      │
├──────────────────────────────────────┤
│  🏠 Home  │ 📹 Videos │ 📊 Jobs │ ⚙️ More │
└──────────────────────────────────────┘
```

### Screen Designs

#### 1. Dashboard (Home)
```
┌──────────────────────────────────┐
│ BeeMonitor              [🔔][👤]│
├──────────────────────────────────┤
│ ┌─────────┐ ┌─────────┐        │
│ │  12      │ │  847    │        │
│ │ Videos   │ │ Events  │        │
│ └─────────┘ └─────────┘        │
│ ┌─────────┐ ┌─────────┐        │
│ │  3       │ │  2      │        │
│ │ Running  │ │ Sources │        │
│ └─────────┘ └─────────┘        │
│                                  │
│ Recent Jobs                      │
│ ┌──────────────────────────────┐│
│ │ 🟢 field_2024-06-15  Done   ││
│ │    847 events • 2m ago      ││
│ ├──────────────────────────────┤│
│ │ 🔄 field_2024-06-14  75%    ││
│ │    Processing...             ││
│ ├──────────────────────────────┤│
│ │ 🟡 field_2024-06-13  Queued ││
│ │    Waiting for GPU           ││
│ └──────────────────────────────┘│
│                                  │
│ [  + Upload Video  ]            │
└──────────────────────────────────┘
```

#### 2. Video Upload (Mobile-Optimized)
```
┌──────────────────────────────────┐
│ ← Upload Video                   │
├──────────────────────────────────┤
│                                  │
│  ┌────────────────────────────┐  │
│  │                            │  │
│  │    📹 Record Video         │  │
│  │    (use device camera)     │  │
│  │                            │  │
│  └────────────────────────────┘  │
│                                  │
│  ┌────────────────────────────┐  │
│  │    📁 Choose from Files    │  │
│  └────────────────────────────┘  │
│                                  │
│  ┌────────────────────────────┐  │
│  │    ☁️  Import from Cloud    │  │
│  │    S3 │ Azure │ GCS │ Drive│  │
│  └────────────────────────────┘  │
│                                  │
│  Upload Progress:                │
│  ┌────────────────────────────┐  │
│  │ field_video.mp4            │  │
│  │ ████████████░░░ 78%  ⏸     │  │
│  │ 156 MB / 200 MB • 2:30 left│  │
│  └────────────────────────────┘  │
│                                  │
│  ⚡ Offline Mode:                │
│  Videos will upload when you     │
│  have a connection.              │
└──────────────────────────────────┘
```

#### 3. Results Viewer (Swipeable)
```
┌──────────────────────────────────┐
│ ← Results: field_2024-06-15      │
├──────────────────────────────────┤
│ [Summary] [Events] [Chart] [Video]│
├──────────────────────────────────┤
│                                  │
│ Summary                          │
│ ┌──────────────────────────────┐│
│ │ Total Events:     847        ││
│ │ Entries:          423        ││
│ │ Exits:            424        ││
│ │ Unique Tracks:    312        ││
│ │ Active Nests:     48/60      ││
│ │ Peak Hour:        10:00-11:00││
│ └──────────────────────────────┘│
│                                  │
│ Activity by Hour                 │
│ ┌──────────────────────────────┐│
│ │  █                           ││
│ │  █ █                         ││
│ │  █ █ █                       ││
│ │  █ █ █ █ █                   ││
│ │ 8 9 10 11 12 1 2 3 4 5      ││
│ └──────────────────────────────┘│
│                                  │
│ [ Download CSV ] [ Share ]       │
└──────────────────────────────────┘
```

---

## Offline Capabilities

### What Works Offline
- View dashboard with cached data
- Browse previously loaded videos and results
- View cached event data and charts
- Queue video uploads (auto-sync when online)
- Queue new analysis jobs (auto-submit when online)
- Change settings and preferences
- View API documentation

### What Requires Online
- Initial login / registration
- Actual video upload transfer
- GPU processing (runs on Modal)
- Real-time job status updates
- Connecting new data sources
- First-time data loading

### Sync Strategy
```
Online → Offline:
  1. Cache all viewed pages and API responses
  2. Pre-fetch recent job results to IndexedDB
  3. Download event CSVs for offline viewing

Offline → Online (Background Sync):
  1. Upload queued videos (resume-capable, chunked)
  2. Submit queued analysis jobs
  3. Sync settings changes
  4. Refresh cached data
```

---

## Push Notifications

```python
# Django: web-push notifications
from webpush import send_user_notification

def notify_job_complete(user, job):
    send_user_notification(
        user=user,
        payload={
            "head": "Analysis Complete ✓",
            "body": f"{job.video.title}: {job.result.total_events} events detected",
            "url": f"/analysis/{job.id}/results/",
            "icon": "/static/icons/icon-192.png"
        },
        ttl=86400
    )
```

---

## Performance Targets

| Metric | Target |
|--------|--------|
| First Contentful Paint | < 1.5s |
| Largest Contentful Paint | < 2.5s |
| Time to Interactive | < 3.0s |
| Lighthouse PWA Score | > 95 |
| Offline page load | < 500ms |
| Cache size (app shell) | < 5 MB |

---

## Django Integration Points

The PWA is served from the **same Django project** — no separate frontend build:

1. **Manifest:** Served by Django view at `/manifest.json`
2. **Service Worker:** Served from `/sw.js` (root scope)
3. **Templates:** Same Django templates, enhanced with HTMX for reactivity
4. **API:** DRF endpoints consumed by both web and PWA
5. **Auth:** Django sessions + "Remember Me" for persistent login
6. **Static:** Workbox precaching of Django static files

```python
# config/urls.py
urlpatterns = [
    path('manifest.json', views.manifest, name='manifest'),
    path('sw.js', views.service_worker, name='sw'),
    path('offline/', views.offline_page, name='offline'),
    ...
]
```
