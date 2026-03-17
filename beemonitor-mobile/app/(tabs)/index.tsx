import { View, Text, StyleSheet, ScrollView, RefreshControl, TouchableOpacity } from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { apiClient } from '../../src/api/client';
import { listVideos } from '../../src/api/videos';
import { listJobs } from '../../src/api/jobs';
import { OfflineBanner } from '../../src/components/OfflineBanner';

interface StatCardProps {
  title: string;
  value: string | number;
  color: string;
  onPress?: () => void;
}

function StatCard({ title, value, color, onPress }: StatCardProps) {
  return (
    <TouchableOpacity style={[styles.card, { borderLeftColor: color }]} onPress={onPress}>
      <Text style={styles.cardValue}>{value}</Text>
      <Text style={styles.cardTitle}>{title}</Text>
    </TouchableOpacity>
  );
}

export default function DashboardScreen() {
  const router = useRouter();

  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const res = await apiClient.get('/health');
      return res.data;
    },
  });

  const videosQuery = useQuery({
    queryKey: ['videos'],
    queryFn: listVideos,
  });

  const jobsQuery = useQuery({
    queryKey: ['jobs'],
    queryFn: listJobs,
  });

  const isRefreshing =
    healthQuery.isRefetching || videosQuery.isRefetching || jobsQuery.isRefetching;

  const onRefresh = () => {
    healthQuery.refetch();
    videosQuery.refetch();
    jobsQuery.refetch();
  };

  const videos = videosQuery.data ?? [];
  const jobs = jobsQuery.data ?? [];
  const activeJobs = jobs.filter(
    (j: any) => j.status === 'processing' || j.status === 'queued'
  );
  const completedJobs = jobs.filter((j: any) => j.status === 'completed');

  return (
    <ScrollView
      style={styles.container}
      refreshControl={<RefreshControl refreshing={isRefreshing} onRefresh={onRefresh} />}
    >
      <OfflineBanner />
      <Text style={styles.header}>Dashboard</Text>
      <Text style={styles.status}>
        API: {healthQuery.isLoading ? 'Checking...' : healthQuery.isError ? 'Offline' : 'Online'}
      </Text>

      <View style={styles.grid}>
        <StatCard
          title="Videos"
          value={videos.length}
          color="#3B82F6"
          onPress={() => router.push('/(tabs)/videos')}
        />
        <StatCard
          title="Completed Jobs"
          value={completedJobs.length}
          color="#10B981"
        />
        <StatCard
          title="Active Jobs"
          value={activeJobs.length}
          color="#F59E0B"
          onPress={() => router.push('/(tabs)/jobs')}
        />
        <StatCard
          title="Total Jobs"
          value={jobs.length}
          color="#8B5CF6"
        />
      </View>

      <TouchableOpacity
        style={styles.uploadButton}
        onPress={() => router.push('/video/upload')}
      >
        <Text style={styles.uploadButtonText}>+ Upload Video</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.newJobButton}
        onPress={() => router.push('/job/new')}
      >
        <Text style={styles.newJobButtonText}>+ New Analysis Job</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F3F4F6', padding: 16 },
  header: { fontSize: 28, fontWeight: 'bold', color: '#111827', marginBottom: 4 },
  status: { fontSize: 14, color: '#6B7280', marginBottom: 20 },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  card: {
    width: '48%',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  cardValue: { fontSize: 28, fontWeight: 'bold', color: '#111827' },
  cardTitle: { fontSize: 13, color: '#6B7280', marginTop: 4 },
  uploadButton: {
    backgroundColor: '#F59E0B',
    borderRadius: 10,
    padding: 16,
    alignItems: 'center',
    marginTop: 8,
  },
  uploadButtonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  newJobButton: {
    backgroundColor: '#3B82F6',
    borderRadius: 10,
    padding: 16,
    alignItems: 'center',
    marginTop: 12,
    marginBottom: 32,
  },
  newJobButtonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
