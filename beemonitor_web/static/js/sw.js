/**
 * BeeMonitor Service Worker
 *
 * Caching strategies:
 * - App shell (CSS, JS, icons): Cache-first
 * - API calls: Network-first with cache fallback
 * - Page navigation: Network-first, offline fallback page
 */

const CACHE_NAME = "beemonitor-v1";
const OFFLINE_URL = "/offline/";

const PRECACHE_URLS = [
  "/",
  "/offline/",
];

// Install: precache app shell
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_URLS))
  );
  self.skipWaiting();
});

// Activate: clean old caches
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

// Fetch: network-first for pages, cache-first for static
self.addEventListener("fetch", (event) => {
  const { request } = event;

  // Skip non-GET requests
  if (request.method !== "GET") return;

  // API calls: network-first with cache fallback
  if (request.url.includes("/api/")) {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
          return response;
        })
        .catch(() => caches.match(request))
    );
    return;
  }

  // Page navigation: network-first, offline fallback
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
          return response;
        })
        .catch(() =>
          caches.match(request).then((cached) => cached || caches.match(OFFLINE_URL))
        )
    );
    return;
  }

  // Static assets: cache-first
  event.respondWith(
    caches.match(request).then((cached) => cached || fetch(request))
  );
});

// Background Sync: queue uploads for when online
self.addEventListener("sync", (event) => {
  if (event.tag === "video-upload") {
    event.waitUntil(processUploadQueue());
  }
  if (event.tag === "job-submit") {
    event.waitUntil(processJobQueue());
  }
});

async function processUploadQueue() {
  // Placeholder — will be implemented with IndexedDB queue
  console.log("[SW] Processing upload queue");
}

async function processJobQueue() {
  console.log("[SW] Processing job submission queue");
}

// Push notifications
self.addEventListener("push", (event) => {
  const data = event.data ? event.data.json() : {};
  const title = data.head || "BeeMonitor";
  const options = {
    body: data.body || "You have a notification",
    icon: "/static/icons/icon-192.png",
    badge: "/static/icons/icon-192.png",
    data: { url: data.url || "/" },
  };
  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const url = event.notification.data.url;
  event.waitUntil(clients.openWindow(url));
});
