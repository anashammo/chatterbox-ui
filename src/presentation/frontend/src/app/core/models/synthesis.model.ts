/**
 * Embedded voice reference information in synthesis response
 */
export interface VoiceReferenceInfo {
  id: string;
  name: string;
  language: string | null;
}

/**
 * Synthesis model representing a TTS synthesis request
 */
export interface Synthesis {
  id: string;
  input_text: string;
  text_length: number;
  model: string;
  status: SynthesisStatus;
  language: string | null;
  voice_reference_id: string | null;
  voice_reference: VoiceReferenceInfo | null;
  cfg_weight: number;
  exaggeration: number;
  output_file_path: string | null;
  output_duration_seconds: number | null;
  error_message: string | null;
  processing_time_seconds: number | null;
  created_at: string;
  completed_at: string | null;
}

export type SynthesisStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface SynthesisListResponse {
  syntheses: Synthesis[];
  total: number;
  limit: number;
  offset: number;
}

export interface SynthesisCreateRequest {
  text: string;
  model: string;
  language?: string;
  voice_reference_id?: string;
  cfg_weight: number;
  exaggeration: number;
}

export interface TTSModel {
  name: string;
  display_name: string;
  parameters: string;
  description: string;
  supports_voice_cloning: boolean;
  supports_multilingual: boolean;
  supports_paralinguistics: boolean;
  is_loaded: boolean;
}

export interface ModelListResponse {
  models: TTSModel[];
}
