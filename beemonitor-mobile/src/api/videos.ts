import { apiClient } from './client';

export interface Video {
  id: number;
  title?: string;
  filename?: string;
  status?: string;
  file_size?: number;
  duration?: number;
  resolution?: string;
  created_at?: string;
  updated_at?: string;
}

export async function listVideos(): Promise<Video[]> {
  const response = await apiClient.get<Video[]>('/videos');
  return response.data;
}

export async function getVideo(id: string): Promise<Video> {
  const response = await apiClient.get<Video>(`/videos/${id}`);
  return response.data;
}

export async function uploadVideo(file: {
  uri: string;
  name: string;
}): Promise<Video> {
  const formData = new FormData();
  formData.append('file', {
    uri: file.uri,
    name: file.name,
    type: 'video/mp4',
  } as any);

  const response = await apiClient.post<Video>('/videos/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  });
  return response.data;
}
