import { View, Text, StyleSheet, ScrollView, ActivityIndicator, TouchableOpacity } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useQuery } from '@tanstack/react-query';
import { getVideo } from '../../src/api/videos';
import { formatBytes, formatDate, formatDuration } from '../../src/utils/formatters';

export default function VideoDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();

  const { data: video, isLoading, isError } = useQuery({
    queryKey: ['video', id],
    queryFn: () => getVideo(id!),
    enabled: !!id,
  });

  if (isLoading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#F59E0B" />
      </View>
    );
  }

  if (isError || !video) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>Failed to load video.</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <View style={styles.preview}>
        <Text style={styles.previewIcon}>▶</Text>
      </View>

      <View style={styles.info}>
        <Text style={styles.title}>{video.title || video.filename || 'Untitled'}</Text>

        <View style={styles.row}>
          <Text style={styles.label}>Status</Text>
          <Text style={styles.value}>{video.status || 'uploaded'}</Text>
        </View>

        {video.file_size && (
          <View style={styles.row}>
            <Text style={styles.label}>Size</Text>
            <Text style={styles.value}>{formatBytes(video.file_size)}</Text>
          </View>
        )}

        {video.duration && (
          <View style={styles.row}>
            <Text style={styles.label}>Duration</Text>
            <Text style={styles.value}>{formatDuration(video.duration)}</Text>
          </View>
        )}

        {video.created_at && (
          <View style={styles.row}>
            <Text style={styles.label}>Uploaded</Text>
            <Text style={styles.value}>{formatDate(video.created_at)}</Text>
          </View>
        )}

        {video.resolution && (
          <View style={styles.row}>
            <Text style={styles.label}>Resolution</Text>
            <Text style={styles.value}>{video.resolution}</Text>
          </View>
        )}
      </View>

      <TouchableOpacity
        style={styles.analyzeButton}
        onPress={() => router.push({ pathname: '/job/new', params: { videoId: id } })}
      >
        <Text style={styles.analyzeButtonText}>Analyze This Video</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F3F4F6' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  errorText: { fontSize: 16, color: '#EF4444' },
  preview: {
    height: 220,
    backgroundColor: '#1F2937',
    justifyContent: 'center',
    alignItems: 'center',
  },
  previewIcon: { fontSize: 48, color: '#fff' },
  info: {
    backgroundColor: '#fff',
    margin: 16,
    borderRadius: 12,
    padding: 16,
  },
  title: { fontSize: 20, fontWeight: 'bold', color: '#111827', marginBottom: 16 },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: '#F3F4F6',
  },
  label: { fontSize: 14, color: '#6B7280' },
  value: { fontSize: 14, fontWeight: '500', color: '#111827' },
  analyzeButton: {
    backgroundColor: '#3B82F6',
    borderRadius: 10,
    padding: 16,
    alignItems: 'center',
    marginHorizontal: 16,
    marginBottom: 32,
  },
  analyzeButtonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
