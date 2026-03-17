import { useState, useEffect, useCallback } from 'react';

interface OfflineState {
  isOnline: boolean;
  checkConnection: () => Promise<boolean>;
}

export function useOffline(): OfflineState {
  const [isOnline, setIsOnline] = useState(true);

  const checkConnection = useCallback(async (): Promise<boolean> => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      await fetch('https://httpbin.org/get', {
        method: 'HEAD',
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      setIsOnline(true);
      return true;
    } catch {
      setIsOnline(false);
      return false;
    }
  }, []);

  useEffect(() => {
    checkConnection();

    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, [checkConnection]);

  return { isOnline, checkConnection };
}
