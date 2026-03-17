import { View, Text, FlatList, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { listJobs } from '../../src/api/jobs';
import { JobStatusBadge } from '../../src/components/JobStatusBadge';
import { EmptyState } from '../../src/components/EmptyState';
import { OfflineBanner } from '../../src/components/OfflineBanner';
import { formatDate } from '../../src/utils/formatters';

export default function JobsScreen() {
  const router = useRouter();
  const { data: jobs, isLoading, isRefetching, refetch } = useQuery({
    queryKey: ['jobs'],
    queryFn: listJobs,
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
        data={jobs ?? []}
        keyExtractor={(item: any) => String(item.id)}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={styles.card}
            onPress={() => router.push(`/job/${item.id}`)}
          >
            <View style={styles.cardHeader}>
              <Text style={styles.cardTitle} numberOfLines={1}>
                Job #{item.id}
              </Text>
              <JobStatusBadge status={item.status} />
            </View>
            {item.video_title && (
              <Text style={styles.cardSubtitle} numberOfLines={1}>
                Video: {item.video_title}
              </Text>
            )}
            <Text style={styles.cardDate}>{formatDate(item.created_at)}</Text>
          </TouchableOpacity>
        )}
        contentContainerStyle={styles.list}
        refreshing={isRefetching}
        onRefresh={refetch}
        ListEmptyComponent={
          <EmptyState icon="job" message="No jobs yet. Create one to analyze your videos!" />
        }
        ListHeaderComponent={
          <TouchableOpacity
            style={styles.newButton}
            onPress={() => router.push('/job/new')}
          >
            <Text style={styles.newButtonText}>+ New Job</Text>
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
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  cardTitle: { fontSize: 16, fontWeight: '600', color: '#111827', flex: 1, marginRight: 8 },
  cardSubtitle: { fontSize: 14, color: '#6B7280', marginBottom: 4 },
  cardDate: { fontSize: 12, color: '#9CA3AF' },
  newButton: {
    backgroundColor: '#3B82F6',
    borderRadius: 10,
    padding: 14,
    alignItems: 'center',
    marginBottom: 16,
  },
  newButtonText: { color: '#fff', fontSize: 15, fontWeight: '600' },
});
