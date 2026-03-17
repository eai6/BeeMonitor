import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

interface EmptyStateProps {
  icon?: 'video' | 'job' | 'search' | 'generic';
  message: string;
}

const ICONS: Record<string, string> = {
  video: 'No videos',
  job: 'No jobs',
  search: 'No results',
  generic: 'Nothing here',
};

export function EmptyState({ icon = 'generic', message }: EmptyStateProps) {
  return (
    <View style={styles.container}>
      <View style={styles.iconCircle}>
        <Text style={styles.iconText}>
          {icon === 'video' ? 'V' : icon === 'job' ? 'J' : icon === 'search' ? '?' : '-'}
        </Text>
      </View>
      <Text style={styles.message}>{message}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    paddingVertical: 48,
    paddingHorizontal: 32,
  },
  iconCircle: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#F3F4F6',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  iconText: {
    fontSize: 24,
    color: '#9CA3AF',
    fontWeight: 'bold',
  },
  message: {
    fontSize: 15,
    color: '#6B7280',
    textAlign: 'center',
    lineHeight: 22,
  },
});
