import { View, Text, StyleSheet, ScrollView, ActivityIndicator } from 'react-native';
import { useLocalSearchParams } from 'expo-router';
import { useQuery } from '@tanstack/react-query';
import { getJobResults } from '../../../src/api/jobs';

export default function JobResultsScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();

  const { data: results, isLoading, isError } = useQuery({
    queryKey: ['jobResults', id],
    queryFn: () => getJobResults(id!),
    enabled: !!id,
  });

  if (isLoading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#F59E0B" />
      </View>
    );
  }

  if (isError || !results) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>Failed to load results.</Text>
      </View>
    );
  }

  const summary = results.summary || {};
  const events = results.events || [];

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Analysis Results</Text>
      <Text style={styles.subtitle}>Job #{id}</Text>

      <View style={styles.summaryCard}>
        <Text style={styles.sectionTitle}>Summary</Text>

        <View style={styles.statRow}>
          <View style={styles.stat}>
            <Text style={styles.statValue}>{summary.total_events ?? events.length}</Text>
            <Text style={styles.statLabel}>Events Detected</Text>
          </View>
          <View style={styles.stat}>
            <Text style={styles.statValue}>{summary.bee_count ?? '-'}</Text>
            <Text style={styles.statLabel}>Bee Count</Text>
          </View>
        </View>

        <View style={styles.statRow}>
          <View style={styles.stat}>
            <Text style={styles.statValue}>{summary.confidence ?? '-'}</Text>
            <Text style={styles.statLabel}>Avg Confidence</Text>
          </View>
          <View style={styles.stat}>
            <Text style={styles.statValue}>{summary.duration ?? '-'}</Text>
            <Text style={styles.statLabel}>Duration (s)</Text>
          </View>
        </View>
      </View>

      {events.length > 0 && (
        <View style={styles.eventsCard}>
          <Text style={styles.sectionTitle}>Events ({events.length})</Text>
          {events.slice(0, 50).map((event: any, index: number) => (
            <View key={index} style={styles.eventRow}>
              <View style={styles.eventDot} />
              <View style={styles.eventContent}>
                <Text style={styles.eventType}>{event.type || 'detection'}</Text>
                <Text style={styles.eventDetail}>
                  Frame {event.frame ?? '-'} | Confidence: {event.confidence?.toFixed(2) ?? '-'}
                </Text>
              </View>
            </View>
          ))}
          {events.length > 50 && (
            <Text style={styles.moreText}>
              ... and {events.length - 50} more events
            </Text>
          )}
        </View>
      )}

      {results.raw && (
        <View style={styles.rawCard}>
          <Text style={styles.sectionTitle}>Raw Data</Text>
          <Text style={styles.rawText}>
            {JSON.stringify(results.raw, null, 2).slice(0, 2000)}
          </Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F3F4F6', padding: 16 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  errorText: { fontSize: 16, color: '#EF4444' },
  title: { fontSize: 24, fontWeight: 'bold', color: '#111827' },
  subtitle: { fontSize: 14, color: '#6B7280', marginBottom: 20 },
  summaryCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#111827',
    marginBottom: 12,
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 12,
  },
  stat: { alignItems: 'center', flex: 1 },
  statValue: { fontSize: 24, fontWeight: 'bold', color: '#F59E0B' },
  statLabel: { fontSize: 12, color: '#6B7280', marginTop: 4 },
  eventsCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  eventRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  eventDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#F59E0B',
    marginTop: 5,
    marginRight: 10,
  },
  eventContent: { flex: 1 },
  eventType: { fontSize: 14, fontWeight: '500', color: '#111827' },
  eventDetail: { fontSize: 12, color: '#6B7280', marginTop: 2 },
  moreText: { fontSize: 13, color: '#9CA3AF', textAlign: 'center', marginTop: 8 },
  rawCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 32,
  },
  rawText: {
    fontSize: 11,
    color: '#374151',
    fontFamily: 'monospace',
    backgroundColor: '#F9FAFB',
    padding: 8,
    borderRadius: 4,
  },
});
