import { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  Switch,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { listVideos } from '../../src/api/videos';
import { createJob } from '../../src/api/jobs';

export default function NewJobScreen() {
  const { videoId: preselectedVideoId } = useLocalSearchParams<{ videoId?: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();

  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(
    preselectedVideoId ?? null
  );
  const [detectBees, setDetectBees] = useState(true);
  const [countBees, setCountBees] = useState(true);
  const [trackMovement, setTrackMovement] = useState(false);

  const { data: videos, isLoading: videosLoading } = useQuery({
    queryKey: ['videos'],
    queryFn: listVideos,
  });

  const createMutation = useMutation({
    mutationFn: (params: { videoId: string; config: any }) =>
      createJob(params.videoId, params.config),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
      Alert.alert('Success', 'Job created successfully!', [
        { text: 'View Job', onPress: () => router.replace(`/job/${data.id}`) },
      ]);
    },
    onError: (error: any) => {
      const message = error?.response?.data?.detail || 'Failed to create job.';
      Alert.alert('Error', message);
    },
  });

  const handleCreate = () => {
    if (!selectedVideoId) {
      Alert.alert('Error', 'Please select a video.');
      return;
    }

    const config = {
      detect_bees: detectBees,
      count_bees: countBees,
      track_movement: trackMovement,
    };

    createMutation.mutate({ videoId: selectedVideoId, config });
  };

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.sectionTitle}>Select Video</Text>

      {videosLoading ? (
        <ActivityIndicator size="small" color="#F59E0B" />
      ) : (
        <View style={styles.videoList}>
          {(videos ?? []).map((video: any) => (
            <TouchableOpacity
              key={video.id}
              style={[
                styles.videoItem,
                selectedVideoId === String(video.id) && styles.videoItemSelected,
              ]}
              onPress={() => setSelectedVideoId(String(video.id))}
            >
              <Text
                style={[
                  styles.videoName,
                  selectedVideoId === String(video.id) && styles.videoNameSelected,
                ]}
                numberOfLines={1}
              >
                {video.title || video.filename || `Video #${video.id}`}
              </Text>
              {selectedVideoId === String(video.id) && (
                <Text style={styles.checkmark}>OK</Text>
              )}
            </TouchableOpacity>
          ))}
          {(videos ?? []).length === 0 && (
            <Text style={styles.emptyText}>No videos available. Upload one first.</Text>
          )}
        </View>
      )}

      <Text style={[styles.sectionTitle, { marginTop: 24 }]}>Configuration</Text>

      <View style={styles.configCard}>
        <View style={styles.configRow}>
          <Text style={styles.configLabel}>Detect Bees</Text>
          <Switch
            value={detectBees}
            onValueChange={setDetectBees}
            trackColor={{ true: '#F59E0B' }}
          />
        </View>
        <View style={styles.configRow}>
          <Text style={styles.configLabel}>Count Bees</Text>
          <Switch
            value={countBees}
            onValueChange={setCountBees}
            trackColor={{ true: '#F59E0B' }}
          />
        </View>
        <View style={[styles.configRow, { borderBottomWidth: 0 }]}>
          <Text style={styles.configLabel}>Track Movement</Text>
          <Switch
            value={trackMovement}
            onValueChange={setTrackMovement}
            trackColor={{ true: '#F59E0B' }}
          />
        </View>
      </View>

      <TouchableOpacity
        style={[styles.createButton, createMutation.isPending && styles.buttonDisabled]}
        onPress={handleCreate}
        disabled={createMutation.isPending}
      >
        {createMutation.isPending ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.createButtonText}>Create Job</Text>
        )}
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F3F4F6', padding: 16 },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 12,
  },
  videoList: {
    backgroundColor: '#fff',
    borderRadius: 12,
    overflow: 'hidden',
  },
  videoItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 14,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  videoItemSelected: {
    backgroundColor: '#FFFBEB',
  },
  videoName: { fontSize: 15, color: '#111827', flex: 1, marginRight: 8 },
  videoNameSelected: { fontWeight: '600', color: '#F59E0B' },
  checkmark: { fontSize: 14, fontWeight: '700', color: '#F59E0B' },
  emptyText: { padding: 16, color: '#9CA3AF', textAlign: 'center' },
  configCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    paddingHorizontal: 16,
  },
  configRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  configLabel: { fontSize: 15, color: '#111827' },
  createButton: {
    backgroundColor: '#3B82F6',
    borderRadius: 10,
    padding: 16,
    alignItems: 'center',
    marginTop: 24,
    marginBottom: 32,
  },
  buttonDisabled: { opacity: 0.5 },
  createButtonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
