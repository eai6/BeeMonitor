import { View, Text, FlatList, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { listVideos } from '../../src/api/videos';
import { VideoCard } from '../../src/components/VideoCard';
import { EmptyState } from '../../src/components/EmptyState';
import { OfflineBanner } from '../../src/components/OfflineBanner';

export default function VideosScreen() {
  const router = useRouter();
  const { data: videos, isLoading, isRefetching, refetch } = useQuery({
    queryKey: ['videos'],
    queryFn: listVideos,
  });

  if (isLoading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#F59E0B" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <OfflineBanner />
      <FlatList
        data={videos ?? []}
        keyExtractor={(item: any) => String(item.id)}
        renderItem={({ item }) => (
          <TouchableOpacity onPress={() => router.push(`/video/${item.id}`)}>
            <VideoCard video={item} />
          </TouchableOpacity>
        )}
        contentContainerStyle={styles.list}
        refreshing={isRefetching}
        onRefresh={refetch}
        ListEmptyComponent={
          <EmptyState icon="video" message="No videos yet. Upload one to get started!" />
        }
        ListHeaderComponent={
          <TouchableOpacity
            style={styles.uploadButton}
            onPress={() => router.push('/video/upload')}
          >
            <Text style={styles.uploadButtonText}>+ Upload Video</Text>
          </TouchableOpacity>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F3F4F6' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  list: { padding: 16 },
  uploadButton: {
    backgroundColor: '#F59E0B',
    borderRadius: 10,
    padding: 14,
    alignItems: 'center',
    marginBottom: 16,
  },
  uploadButtonText: { color: '#fff', fontSize: 15, fontWeight: '600' },
});
