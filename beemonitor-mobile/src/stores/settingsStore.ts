import { create } from 'zustand';
import { storage } from '../api/client';

interface SettingsState {
  apiUrl: string;
  autoUpload: boolean;
  wifiOnly: boolean;
  setApiUrl: (url: string) => void;
  setAutoUpload: (value: boolean) => void;
  setWifiOnly: (value: boolean) => void;
}

const DEFAULT_API_URL =
  process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export const useSettingsStore = create<SettingsState>((set) => ({
  apiUrl: storage.getString('settings_api_url') || DEFAULT_API_URL,
  autoUpload: storage.getBoolean('settings_auto_upload') ?? false,
  wifiOnly: storage.getBoolean('settings_wifi_only') ?? true,

  setApiUrl: (url: string) => {
    storage.set('settings_api_url', url);
    set({ apiUrl: url });
  },

  setAutoUpload: (value: boolean) => {
    storage.set('settings_auto_upload', value);
    set({ autoUpload: value });
  },

  setWifiOnly: (value: boolean) => {
    storage.set('settings_wifi_only', value);
    set({ wifiOnly: value });
  },
}));
