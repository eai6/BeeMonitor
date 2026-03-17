import { apiClient } from './client';

export interface Job {
  id: number;
  video_id?: number;
  video_title?: string;
  status: string;
  config?: Record<string, any>;
  error?: string;
  created_at?: string;
  started_at?: string;
  completed_at?: string;
}

export interface JobResults {
  summary?: {
    total_events?: number;
    bee_count?: number;
    confidence?: number;
    duration?: number;
  };
  events?: Array<{
    type?: string;
    frame?: number;
    confidence?: number;
    bbox?: number[];
  }>;
  raw?: any;
}

export async function listJobs(): Promise<Job[]> {
  const response = await apiClient.get<Job[]>('/jobs');
  return response.data;
}

export async function getJob(id: string): Promise<Job> {
  const response = await apiClient.get<Job>(`/jobs/${id}`);
  return response.data;
}

export async function createJob(
  videoId: string,
  config: Record<string, any>
): Promise<Job> {
  const response = await apiClient.post<Job>('/jobs', {
    video_id: Number(videoId),
    config,
  });
  return response.data;
}

export async function getJobResults(id: string): Promise<JobResults> {
  const response = await apiClient.get<JobResults>(`/jobs/${id}/results`);
  return response.data;
}
