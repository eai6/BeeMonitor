import { create } from 'zustand';
import { storage } from '../api/client';

interface User {
  id: number;
  username: string;
  email: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isLoggedIn: boolean;
  login: (user: User, token: string) => void;
  logout: () => void;
  hydrate: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: null,
  isLoggedIn: false,

  login: (user: User, token: string) => {
    storage.set('auth_token', token);
    storage.set('auth_user', JSON.stringify(user));
    set({ user, token, isLoggedIn: true });
  },

  logout: () => {
    storage.delete('auth_token');
    storage.delete('auth_user');
    set({ user: null, token: null, isLoggedIn: false });
  },

  hydrate: () => {
    const token = storage.getString('auth_token');
    const userStr = storage.getString('auth_user');
    if (token && userStr) {
      try {
        const user = JSON.parse(userStr) as User;
        set({ user, token, isLoggedIn: true });
      } catch {
        storage.delete('auth_token');
        storage.delete('auth_user');
      }
    }
  },
}));

// Hydrate on load
useAuthStore.getState().hydrate();
