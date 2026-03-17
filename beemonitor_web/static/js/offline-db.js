/**
 * BeeMonitor Offline Database (IndexedDB via Dexie.js)
 *
 * Stores jobs, results, events, and sync queue for offline support.
 * Include Dexie.js CDN before this script:
 * <script src="https://unpkg.com/dexie/dist/dexie.js"></script>
 */

const db = new Dexie("BeeMonitorOffline");

db.version(1).stores({
  jobs: "++id, serverId, status, videoId, createdAt",
  results: "++id, jobId, eventCount, entryCount, exitCount",
  events: "++id, jobId, nestId, action, timestamp",
  videos: "++id, serverId, title, status, localPath",
  settings: "key",
  syncQueue: "++id, action, endpoint, status, createdAt",
});

// Cache API responses for offline viewing
async function cacheJobList(jobs) {
  await db.jobs.clear();
  await db.jobs.bulkAdd(
    jobs.map((j) => ({
      serverId: j.id,
      status: j.status,
      videoId: j.video,
      createdAt: j.created_at,
    }))
  );
}

// Queue an action for when we're back online
async function queueAction(action, endpoint, payload) {
  await db.syncQueue.add({
    action,
    endpoint,
    payload: JSON.stringify(payload),
    status: "pending",
    createdAt: new Date().toISOString(),
  });

  // Request background sync if available
  if ("serviceWorker" in navigator && "SyncManager" in window) {
    const reg = await navigator.serviceWorker.ready;
    await reg.sync.register(action);
  }
}

// Process queued actions when back online
async function processSyncQueue() {
  const pending = await db.syncQueue.where("status").equals("pending").toArray();
  for (const item of pending) {
    try {
      const resp = await fetch(item.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: item.payload,
      });
      if (resp.ok) {
        await db.syncQueue.update(item.id, { status: "completed" });
      }
    } catch {
      // Still offline, leave as pending
    }
  }
}

// Auto-sync when back online
window.addEventListener("online", processSyncQueue);
