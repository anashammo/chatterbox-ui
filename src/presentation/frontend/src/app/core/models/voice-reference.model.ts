/**
 * Voice Reference model for voice cloning
 */
export interface VoiceReference {
  id: string;
  name: string;
  original_filename: string;
  file_size_bytes: number;
  file_size_mb: number;
  mime_type: string;
  duration_seconds: number;
  language?: string;
  created_at: string;
}

export interface VoiceReferenceListResponse {
  voice_references: VoiceReference[];
  total: number;
  limit: number;
  offset: number;
}

export interface VoiceReferenceUploadResponse {
  id: string;
  name: string;
  original_filename: string;
  file_size_mb: number;
  duration_seconds: number;
  language?: string;
  message: string;
}
