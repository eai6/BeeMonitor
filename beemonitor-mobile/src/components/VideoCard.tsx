import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { formatBytes, formatDate } from '../utils/formatters';
import type { Video } from '../api/videos';

interface VideoCardProps {
  video: Video;
}

export function VideoCard({ video }: VideoCardProps) {
  return (
    <View style={styles.card}>
      <View style={styles.thumbnail}>
        <Text style={styles.thumbnailIcon}>V</Text>
      </View>
      <View style={styles.info}>
        <Text style={styles.title} numberOfLines={1}>
          {video.title || video.filename || `Video #${video.id}`}
        </Text>
        <View style={styles.meta}>
          {video.status && (
            <Text style={styles.status}>{video.status}</Text>
          )}
          {video.file_size != null && (
            <Text style={styles.detail}>{formatBytes(video.file_size)}</Text>
          )}
        </View>
        {video.created_at && (
          <Text style={styles.date}>{formatDate(video.created_at)}</Text>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 12,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  thumbnail: {
    width: 56,
    height: 56,
    borderRadius: 8,
    backgroundColor: '#1F2937',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  thumbnailIcon: { color: '#F59E0B', fontSize: 20, fontWeight: 'bold' },
  info: { flex: 1, justifyContent: 'center' },
  title: { fontSize: 15, fontWeight: '600', color: '#111827', marginBottom: 4 },
  meta: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  status: {
    fontSize: 12,
    color: '#6B7280',
    backgroundColor: '#F3F4F6',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    overflow: 'hidden',
  },
  detail: { fontSize: 12, color: '#9CA3AF' },
  date: { fontSize: 11, color: '#9CA3AF', marginTop: 4 },
});
