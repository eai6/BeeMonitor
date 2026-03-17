import { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { useRouter } from 'expo-router';
import * as FileSystem from 'expo-file-system';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { uploadVideo } from '../../src/api/videos';

export default function UploadScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [selectedFile, setSelectedFile] = useState<{
    uri: string;
    name: string;
    size?: number;
  } | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const uploadMutation = useMutation({
    mutationFn: async (file: { uri: string; name: string }) => {
      return uploadVideo(file);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['videos'] });
      Alert.alert('Success', 'Video uploaded successfully!', [
        { text: 'OK', onPress: () => router.back() },
      ]);
    },
    onError: (error: any) => {
      const message = error?.response?.data?.detail || 'Upload failed. Please try again.';
      Alert.alert('Upload Failed', message);
    },
  });

  const pickFile = async () => {
    // In a real app, use expo-image-picker or expo-document-picker
    // For now, show a placeholder
    Alert.alert(
      'Pick Video',
      'In production, this would open the gallery picker. For now, using a mock file.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Use Mock',
          onPress: () => {
            setSelectedFile({
              uri: FileSystem.documentDirectory + 'sample.mp4',
              name: 'sample.mp4',
              size: 1024 * 1024 * 10,
            });
          },
        },
      ]
    );
  };

  const handleUpload = () => {
    if (!selectedFile) {
      Alert.alert('Error', 'Please select a video first.');
      return;
    }
    setUploadProgress(0);
    uploadMutation.mutate(selectedFile);
  };

  return (
    <View style={styles.container}>
      <View style={styles.dropZone}>
        {selectedFile ? (
          <View style={styles.fileInfo}>
            <Text style={styles.fileIcon}>V</Text>
            <Text style={styles.fileName}>{selectedFile.name}</Text>
            {selectedFile.size && (
              <Text style={styles.fileSize}>
                {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
              </Text>
            )}
          </View>
        ) : (
          <View style={styles.placeholder}>
            <Text style={styles.placeholderIcon}>+</Text>
            <Text style={styles.placeholderText}>Select a video to upload</Text>
          </View>
        )}
      </View>

      <TouchableOpacity style={styles.pickButton} onPress={pickFile}>
        <Text style={styles.pickButtonText}>
          {selectedFile ? 'Change Video' : 'Pick from Gallery'}
        </Text>
      </TouchableOpacity>

      {uploadMutation.isPending && (
        <View style={styles.progressContainer}>
          <View style={styles.progressBar}>
            <View style={[styles.progressFill, { width: `${uploadProgress}%` }]} />
          </View>
          <Text style={styles.progressText}>Uploading...</Text>
        </View>
      )}

      <TouchableOpacity
        style={[
          styles.uploadButton,
          (!selectedFile || uploadMutation.isPending) && styles.buttonDisabled,
        ]}
        onPress={handleUpload}
        disabled={!selectedFile || uploadMutation.isPending}
      >
        {uploadMutation.isPending ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.uploadButtonText}>Upload Video</Text>
        )}
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F3F4F6', padding: 16 },
  dropZone: {
    backgroundColor: '#fff',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#E5E7EB',
    borderStyle: 'dashed',
    padding: 32,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 200,
  },
  placeholder: { alignItems: 'center' },
  placeholderIcon: { fontSize: 48, color: '#D1D5DB' },
  placeholderText: { fontSize: 16, color: '#9CA3AF', marginTop: 8 },
  fileInfo: { alignItems: 'center' },
  fileIcon: { fontSize: 40, color: '#F59E0B', marginBottom: 8 },
  fileName: { fontSize: 16, fontWeight: '600', color: '#111827' },
  fileSize: { fontSize: 14, color: '#6B7280', marginTop: 4 },
  pickButton: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 14,
    alignItems: 'center',
    marginTop: 16,
    borderWidth: 1,
    borderColor: '#D1D5DB',
  },
  pickButtonText: { color: '#374151', fontSize: 15, fontWeight: '500' },
  progressContainer: { marginTop: 16 },
  progressBar: {
    height: 6,
    backgroundColor: '#E5E7EB',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#F59E0B',
    borderRadius: 3,
  },
  progressText: { fontSize: 13, color: '#6B7280', textAlign: 'center', marginTop: 6 },
  uploadButton: {
    backgroundColor: '#F59E0B',
    borderRadius: 10,
    padding: 16,
    alignItems: 'center',
    marginTop: 20,
  },
  buttonDisabled: { opacity: 0.5 },
  uploadButtonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
