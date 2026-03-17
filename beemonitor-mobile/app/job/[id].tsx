import { View, Text, StyleSheet, ScrollView, ActivityIndicator, TouchableOpacity, Platform } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useQuery } from '@tanstack/react-query';
import { getJob } from '../../src/api/jobs';
import { JobStatusBadge } from '../../src/components/JobStatusBadge';
import { formatDate } from '../../src/utils/formatters';

export default function JobDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();

  const { data: job, isLoading, isError } = useQuery({
    queryKey: ['job', id],
    queryFn: () => getJob(id!),
    enabled: !!id,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === 'queued' || status === 'processing') return 5000;
      return false;
    },
  });

  if (isLoading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#F59E0B" />
      </View>
    );
  }

  if (isError || !job) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>Failed to load job.</Text>
      </View>
    );
  }

  const isActive = job.status === 'queued' || job.status === 'processing';

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Job #{job.id}</Text>
        <JobStatusBadge status={job.status} />
      </View>

      {isActive && (
        <View style={styles.activeIndicator}>
          <ActivityIndicator size="small" color="#3B82F6" />
          <Text style={styles.activeText}>
            {job.status === 'queued' ? 'Waiting in queue...' : 'Processing...'}
          </Text>
        </View>
      )}

      <View style={styles.card}>
        <View style={styles.row}>
          <Text style={styles.label}>Status</Text>
          <Text style={styles.value}>{job.status}</Text>
        </View>

        {job.video_id && (
          <View style={styles.row}>
            <Text style={styles.label}>Video ID</Text>
            <Text style={styles.value}>{job.video_id}</Text>
          </View>
        )}

        {job.created_at && (
          <View style={styles.row}>
            <Text style={styles.label}>Created</Text>
            <Text style={styles.value}>{formatDate(job.created_at)}</Text>
          </View>
        )}

        {job.started_at && (
          <View style={styles.row}>
            <Text style={styles.label}>Started</Text>
            <Text style={styles.value}>{formatDate(job.started_at)}</Text>
          </View>
        )}

        {job.completed_at && (
          <View style={styles.row}>
            <Text style={styles.label}>Completed</Text>
            <Text style={styles.value}>{formatDate(job.completed_at)}</Text>
          </View>
        )}

        {job.error && (
          <View style={styles.errorRow}>
            <Text style={styles.label}>Error</Text>
            <Text style={styles.errorValue}>{job.error}</Text>
          </View>
        )}

        {job.config && (
          <View style={styles.configRow}>
            <Text style={styles.label}>Configuration</Text>
            <Text style={styles.configValue}>
              {JSON.stringify(job.config, null, 2)}
            </Text>
          </View>
        )}
      </View>

      {job.status === 'completed' && (
        <TouchableOpacity
          style={styles.resultsButton}
          onPress={() => router.push(`/job/results/${job.id}`)}
        >
          <Text style={styles.resultsButtonText}>View Results</Text>
        </TouchableOpacity>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F3F4F6', padding: 16 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  errorText: { fontSize: 16, color: '#EF4444' },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  title: { fontSize: 24, fontWeight: 'bold', color: '#111827' },
  activeIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#EFF6FF',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  activeText: { marginLeft: 8, color: '#3B82F6', fontSize: 14 },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  label: { fontSize: 14, color: '#6B7280' },
  value: { fontSize: 14, fontWeight: '500', color: '#111827' },
  errorRow: {
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  errorValue: { fontSize: 14, color: '#EF4444', marginTop: 4 },
  configRow: {
    paddingVertical: 10,
  },
  configValue: {
    fontSize: 12,
    color: '#374151',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    marginTop: 4,
    backgroundColor: '#F9FAFB',
    padding: 8,
    borderRadius: 4,
  },
  resultsButton: {
    backgroundColor: '#10B981',
    borderRadius: 10,
    padding: 16,
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 32,
  },
  resultsButtonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
