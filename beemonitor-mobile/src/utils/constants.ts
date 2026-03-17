/**
 * Status color definitions for job badges and indicators.
 */
export const STATUS_COLORS: Record<string, { bg: string; text: string }> = {
  queued: { bg: '#FEF3C7', text: '#92400E' },
  processing: { bg: '#DBEAFE', text: '#1E40AF' },
  completed: { bg: '#D1FAE5', text: '#065F46' },
  failed: { bg: '#FEE2E2', text: '#991B1B' },
  cancelled: { bg: '#F3F4F6', text: '#6B7280' },
  unknown: { bg: '#F3F4F6', text: '#6B7280' },
};

/**
 * API endpoint paths (relative to base URL).
 */
export const API_ENDPOINTS = {
  HEALTH: '/health',
  AUTH_LOGIN: '/auth/login',
  AUTH_REGISTER: '/auth/register',
  VIDEOS: '/videos',
  VIDEO_UPLOAD: '/videos/upload',
  JOBS: '/jobs',
} as const;

/**
 * App-wide constants.
 */
export const APP_NAME = 'BeeMonitor';
export const DEFAULT_PAGE_SIZE = 20;
export const JOB_POLL_INTERVAL = 5000;
