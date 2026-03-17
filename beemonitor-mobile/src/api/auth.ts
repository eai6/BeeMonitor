import { apiClient } from './client';

interface LoginResponse {
  access_token: string;
  token_type: string;
  user: {
    id: number;
    username: string;
    email: string;
  };
}

interface RegisterResponse {
  id: number;
  username: string;
  email: string;
}

export async function login(
  username: string,
  password: string
): Promise<LoginResponse> {
  // OAuth2 password flow uses form-encoded body
  const formData = new URLSearchParams();
  formData.append('username', username);
  formData.append('password', password);

  const response = await apiClient.post<LoginResponse>('/auth/login', formData.toString(), {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  });
  return response.data;
}

export async function register(
  username: string,
  email: string,
  password: string
): Promise<RegisterResponse> {
  const response = await apiClient.post<RegisterResponse>('/auth/register', {
    username,
    email,
    password,
  });
  return response.data;
}
