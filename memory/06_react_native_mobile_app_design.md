# BeeMonitor React Native Mobile Application Design

## Overview

A mobile-first, offline-capable React Native app for iOS and Android that provides field researchers with full BeeMonitor functionality — video capture, cloud upload, analysis management, and results viewing — optimized for outdoor field conditions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  REACT NATIVE APP                        │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │              UI Layer (React Native)             │    │
│  │  • React Navigation (tab + stack)               │    │
│  │  • NativeWind (Tailwind for RN)                 │    │
│  │  • React Native Paper (Material UI)             │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────┴──────────────────────────┐    │
│  │            State Management                      │    │
│  │  • Zustand (global state)                       │    │
│  │  • TanStack Query (server state + cache)        │    │
│  │  • MMKV (fast key-value storage)                │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────┴──────────────────────────┐    │
│  │            Offline Layer                          │    │
│  │  • WatermelonDB (SQLite-based offline DB)       │    │
│  │  • Background Upload (react-native-upload)      │    │
│  │  • NetInfo (connectivity detection)             │    │
│  │  • Background Fetch (periodic sync)             │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────┴──────────────────────────┐    │
│  │            Native Modules                         │    │
│  │  • Camera (react-native-vision-camera)          │    │
│  │  • File System (react-native-fs)                │    │
│  │  • Share (react-native-share)                   │    │
│  │  • Push Notifications (Firebase/APNs)           │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│               BEEMONITOR CLOUD API                        │
│  Django REST API → Modal GPU Processing → Azure Storage  │
└──────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Framework | React Native 0.76+ (New Architecture) | Cross-platform, large ecosystem |
| Navigation | React Navigation 7 | Tab + stack navigation |
| Styling | NativeWind (Tailwind) | Consistent with web, utility-first |
| State | Zustand + TanStack Query | Lightweight, great offline support |
| Offline DB | WatermelonDB | SQLite-based, lazy loading, sync |
| Storage | MMKV | Fast key-value (settings, tokens) |
| Camera | Vision Camera v4 | High-perf camera with frame processing |
| Networking | Axios + Background Upload | Chunked upload, retry, resume |
| Charts | Victory Native | Offline-capable charting |
| Testing | Jest + React Native Testing Library | Unit + integration |
| Build | Expo (managed) or bare RN | EAS Build for CI/CD |

---

## Project Structure

```
beemonitor-mobile/
├── app/                          # Expo Router (file-based routing)
│   ├── (tabs)/                   # Tab navigator
│   │   ├── _layout.tsx           # Tab bar configuration
│   │   ├── index.tsx             # Dashboard (Home tab)
│   │   ├── videos.tsx            # Videos tab
│   │   ├── jobs.tsx              # Jobs tab
│   │   └── more.tsx              # Settings/More tab
│   │
│   ├── (auth)/                   # Auth stack
│   │   ├── login.tsx
│   │   ├── register.tsx
│   │   └── forgot-password.tsx
│   │
│   ├── video/
│   │   ├── [id].tsx              # Video detail
│   │   ├── upload.tsx            # Upload screen
│   │   └── capture.tsx           # Camera capture
│   │
│   ├── job/
│   │   ├── [id].tsx              # Job detail
│   │   ├── new.tsx               # New analysis job
│   │   └── results/[id].tsx      # Results viewer
│   │
│   ├── sources/
│   │   ├── index.tsx             # Source list
│   │   └── add.tsx               # Add new source
│   │
│   ├── developer/
│   │   ├── index.tsx             # API keys
│   │   └── docs.tsx              # API docs
│   │
│   └── settings/
│       ├── index.tsx             # Settings list
│       ├── profile.tsx
│       └── notifications.tsx
│
├── src/
│   ├── api/                      # API client
│   │   ├── client.ts             # Axios instance + interceptors
│   │   ├── auth.ts               # Auth endpoints
│   │   ├── videos.ts             # Video endpoints
│   │   ├── jobs.ts               # Job endpoints
│   │   └── sources.ts            # Source endpoints
│   │
│   ├── db/                       # WatermelonDB
│   │   ├── schema.ts             # Database schema
│   │   ├── models/
│   │   │   ├── Video.ts
│   │   │   ├── Job.ts
│   │   │   ├── Event.ts
│   │   │   └── SyncQueue.ts
│   │   └── sync.ts               # Sync logic with server
│   │
│   ├── stores/                   # Zustand stores
│   │   ├── authStore.ts
│   │   ├── uploadStore.ts
│   │   └── settingsStore.ts
│   │
│   ├── hooks/                    # Custom hooks
│   │   ├── useOffline.ts         # Offline detection + queue
│   │   ├── useUpload.ts          # Chunked upload with progress
│   │   ├── useJobPolling.ts      # Real-time job status
│   │   └── useSync.ts            # Background sync
│   │
│   ├── components/               # Shared components
│   │   ├── VideoCard.tsx
│   │   ├── JobStatusBadge.tsx
│   │   ├── UploadProgress.tsx
│   │   ├── EventsTable.tsx
│   │   ├── ActivityChart.tsx
│   │   ├── NestHeatmap.tsx
│   │   ├── OfflineBanner.tsx
│   │   └── EmptyState.tsx
│   │
│   └── utils/
│       ├── formatters.ts
│       ├── constants.ts
│       └── permissions.ts
│
├── assets/
│   ├── icons/
│   └── images/
│
├── app.json                      # Expo config
├── eas.json                      # EAS Build config
└── package.json
```

---

## Offline Database (WatermelonDB)

### Schema

```typescript
// src/db/schema.ts
import { appSchema, tableSchema } from '@nozbe/watermelondb';

export const schema = appSchema({
  version: 1,
  tables: [
    tableSchema({
      name: 'videos',
      columns: [
        { name: 'server_id', type: 'string', isOptional: true },
        { name: 'title', type: 'string' },
        { name: 'local_path', type: 'string', isOptional: true },
        { name: 'azure_path', type: 'string', isOptional: true },
        { name: 'file_size', type: 'number' },
        { name: 'duration', type: 'number', isOptional: true },
        { name: 'resolution', type: 'string', isOptional: true },
        { name: 'status', type: 'string' },  // local, uploading, uploaded, processing
        { name: 'upload_progress', type: 'number' },
        { name: 'created_at', type: 'number' },
        { name: 'synced_at', type: 'number', isOptional: true },
      ],
    }),
    tableSchema({
      name: 'jobs',
      columns: [
        { name: 'server_id', type: 'string', isOptional: true },
        { name: 'video_id', type: 'string' },
        { name: 'status', type: 'string' },
        { name: 'progress', type: 'number' },
        { name: 'config_json', type: 'string' },
        { name: 'total_events', type: 'number', isOptional: true },
        { name: 'entry_count', type: 'number', isOptional: true },
        { name: 'exit_count', type: 'number', isOptional: true },
        { name: 'error_message', type: 'string', isOptional: true },
        { name: 'created_at', type: 'number' },
        { name: 'completed_at', type: 'number', isOptional: true },
      ],
    }),
    tableSchema({
      name: 'events',
      columns: [
        { name: 'job_id', type: 'string' },
        { name: 'nest_id', type: 'number' },
        { name: 'action', type: 'string' },  // entry, exit
        { name: 'timestamp', type: 'string' },
        { name: 'frame_number', type: 'number' },
        { name: 'track_id', type: 'number' },
        { name: 'confidence', type: 'number' },
      ],
    }),
    tableSchema({
      name: 'sync_queue',
      columns: [
        { name: 'action', type: 'string' },     // upload, submit_job, etc.
        { name: 'payload_json', type: 'string' },
        { name: 'status', type: 'string' },      // pending, in_progress, failed
        { name: 'retry_count', type: 'number' },
        { name: 'created_at', type: 'number' },
      ],
    }),
  ],
});
```

---

## Key Screens

### 1. Dashboard

```
┌──────────────────────────────────┐
│ BeeMonitor           [🔔] [👤]  │
├──────────────────────────────────┤
│                                  │
│  ┌─────────┐ ┌─────────┐       │
│  │   12    │ │   847   │       │
│  │ Videos  │ │ Events  │       │
│  └─────────┘ └─────────┘       │
│  ┌─────────┐ ┌─────────┐       │
│  │    3    │ │    2    │       │
│  │ Running │ │ Sources │       │
│  └─────────┘ └─────────┘       │
│                                  │
│  ⚡ Quick Actions                │
│  ┌──────────────────────────┐   │
│  │ 📹 Capture  │ 📤 Upload  │   │
│  └──────────────────────────┘   │
│                                  │
│  Recent Activity                 │
│  ─────────────────────────────  │
│  🟢 field_06-15  847 events     │
│  🔄 field_06-14  Processing 75% │
│  🟡 field_06-13  Queued         │
│                                  │
│  ⚠️ Offline: 2 uploads pending   │
│                                  │
├──────────────────────────────────┤
│ 🏠 Home │ 📹 Video │ 📊 Jobs │ ⋯ │
└──────────────────────────────────┘
```

### 2. Video Capture (Field Use)

```
┌──────────────────────────────────┐
│ ← Capture Video                  │
├──────────────────────────────────┤
│                                  │
│  ┌────────────────────────────┐  │
│  │                            │  │
│  │                            │  │
│  │     [Camera Preview]       │  │
│  │                            │  │
│  │                            │  │
│  │  REC 00:45:23    1080p 30  │  │
│  └────────────────────────────┘  │
│                                  │
│  Site: Field Station A           │
│  Storage: 12.4 GB free           │
│                                  │
│         ┌─────────┐              │
│         │  ⏺ REC  │              │
│         └─────────┘              │
│                                  │
│  [Settings]    [Gallery]         │
│                                  │
│  Auto-upload when WiFi: ✅       │
└──────────────────────────────────┘
```

### 3. Upload Queue (Offline-Aware)

```
┌──────────────────────────────────┐
│ ← Upload Queue                   │
├──────────────────────────────────┤
│                                  │
│  📶 Status: WiFi Connected       │
│  🔄 Auto-uploading...            │
│                                  │
│  ┌────────────────────────────┐  │
│  │ 📤 field_06-16_10_00.mp4   │  │
│  │ ████████████░░░ 78%        │  │
│  │ 156 MB / 200 MB • 2:30     │  │
│  │ [Pause]                    │  │
│  ├────────────────────────────┤  │
│  │ ⏳ field_06-16_11_00.mp4   │  │
│  │ 340 MB • Waiting...        │  │
│  ├────────────────────────────┤  │
│  │ ⏳ field_06-16_12_00.mp4   │  │
│  │ 280 MB • Waiting...        │  │
│  └────────────────────────────┘  │
│                                  │
│  Total queued: 3 files (820 MB)  │
│                                  │
│  Settings:                       │
│  WiFi only upload: ✅            │
│  Auto-analyze after upload: ✅   │
│  Chunk size: 5 MB                │
└──────────────────────────────────┘
```

### 4. Results Viewer

```
┌──────────────────────────────────┐
│ ← field_2024-06-15 Results       │
├──────────────────────────────────┤
│ [Summary│Events│Chart│Nests]     │
├──────────────────────────────────┤
│                                  │
│  ┌──────────────────────────┐   │
│  │ Events: 847              │   │
│  │ Entries: 423 │ Exits: 424│   │
│  │ Tracks: 312  │ Nests: 48 │   │
│  └──────────────────────────┘   │
│                                  │
│  Hourly Activity                 │
│  ┌──────────────────────────┐   │
│  │     ██                    │   │
│  │   ████ ██                 │   │
│  │  ██████████               │   │
│  │ ████████████ ██           │   │
│  │ 8  10  12  2  4  6       │   │
│  └──────────────────────────┘   │
│                                  │
│  Top Active Nests                │
│  Nest 23: ████████████ 45       │
│  Nest 17: ██████████   38       │
│  Nest 42: █████████    34       │
│  Nest 8:  ████████     31       │
│                                  │
│  [📥 Download CSV] [📤 Share]    │
│                                  │
│  Available offline ✅             │
└──────────────────────────────────┘
```

---

## Offline Sync Flow

```
┌─────────────────────────────────────────────┐
│                 ONLINE                       │
│                                              │
│  App Start → Check connectivity              │
│           → Sync: pull server changes        │
│           → Sync: push queued actions        │
│           → Cache: prefetch recent results   │
│                                              │
│  User Action → API call → Update local DB    │
│             → If fails → Queue for later     │
└──────────────────────┬──────────────────────┘
                       │ connection lost
                       ▼
┌─────────────────────────────────────────────┐
│                 OFFLINE                       │
│                                              │
│  Show "Offline" banner                       │
│  All reads from WatermelonDB                 │
│  All writes queued in sync_queue table       │
│                                              │
│  Video capture → Save to local filesystem    │
│  Submit job → Queue in sync_queue            │
│  View results → From cached data             │
│                                              │
│  Background: NetInfo listener waiting        │
└──────────────────────┬──────────────────────┘
                       │ connection restored
                       ▼
┌─────────────────────────────────────────────┐
│              RECONNECTED                     │
│                                              │
│  Process sync_queue (FIFO):                  │
│  1. Upload queued videos (chunked, resume)   │
│  2. Submit queued analysis jobs              │
│  3. Sync settings changes                   │
│                                              │
│  Pull updates:                               │
│  1. Refresh job statuses                     │
│  2. Download new results                     │
│  3. Update dashboard stats                   │
└─────────────────────────────────────────────┘
```

---

## Background Upload System

```typescript
// src/hooks/useUpload.ts
import BackgroundUpload from 'react-native-background-upload';

export function useChunkedUpload() {
  const uploadVideo = async (localPath: string, videoId: string) => {
    const fileSize = await RNFS.stat(localPath).then(s => s.size);
    const CHUNK_SIZE = 5 * 1024 * 1024; // 5 MB
    const totalChunks = Math.ceil(fileSize / CHUNK_SIZE);

    // Get upload session from server
    const { uploadUrl, sessionId } = await api.videos.initUpload(videoId, {
      fileSize, totalChunks, fileName: path.basename(localPath)
    });

    // Upload chunks (resumes from last successful chunk)
    for (let i = lastChunk; i < totalChunks; i++) {
      const chunk = await readChunk(localPath, i * CHUNK_SIZE, CHUNK_SIZE);
      await api.videos.uploadChunk(sessionId, i, chunk);
      updateProgress(videoId, (i + 1) / totalChunks);
    }

    // Finalize
    await api.videos.finalizeUpload(sessionId);
  };

  return { uploadVideo };
}
```

---

## Push Notifications

```typescript
// Firebase Cloud Messaging setup
import messaging from '@react-native-firebase/messaging';

// Notification types:
// - job_complete: Analysis finished
// - job_failed: Analysis error
// - upload_complete: Video uploaded (background)
// - system: Maintenance, updates

messaging().onMessage(async remoteMessage => {
  const { type, jobId, title, body } = remoteMessage.data;

  if (type === 'job_complete') {
    // Update local DB
    await db.jobs.find(jobId).update(j => { j.status = 'completed'; });
    // Prefetch results
    await syncJobResults(jobId);
    // Show local notification
    showNotification({ title, body, deepLink: `/job/results/${jobId}` });
  }
});
```

---

## Field-Specific Features

| Feature | Implementation |
|---------|---------------|
| Low bandwidth mode | Compress uploads, skip annotated video download |
| Battery optimization | Pause uploads on low battery, reduce sync frequency |
| GPS tagging | Auto-tag videos with location (field site mapping) |
| Offline capture | Record hours of video, queue for upload when WiFi available |
| Multi-site management | Organize videos by site, each with own nest config |
| Quick field notes | Attach text/voice notes to videos before upload |
| Solar power awareness | Schedule uploads during peak solar hours |

---

## Build & Distribution

```json
// eas.json
{
  "build": {
    "development": {
      "distribution": "internal",
      "ios": { "simulator": true }
    },
    "preview": {
      "distribution": "internal"
    },
    "production": {
      "ios": { "buildConfiguration": "Release" },
      "android": { "buildType": "app-bundle" }
    }
  },
  "submit": {
    "production": {
      "ios": { "appleId": "...", "ascAppId": "..." },
      "android": { "serviceAccountKeyPath": "./google-services.json" }
    }
  }
}
```

**Distribution:**
- iOS: TestFlight → App Store
- Android: Internal testing → Play Store
- Direct APK: For field devices without Play Store
