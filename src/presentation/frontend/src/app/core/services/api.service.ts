import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';
import { Synthesis, SynthesisListResponse, SynthesisCreateRequest, ModelListResponse } from '../models/synthesis.model';
import { VoiceReference, VoiceReferenceListResponse, VoiceReferenceUploadResponse } from '../models/voice-reference.model';

/**
 * API service for HTTP communication with Chatterbox TTS backend
 */
@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private readonly apiUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  // ========== Synthesis Endpoints ==========

  /**
   * Create a new TTS synthesis
   */
  createSynthesis(request: SynthesisCreateRequest): Observable<Synthesis> {
    return this.http.post<Synthesis>(
      `${this.apiUrl}/syntheses`,
      request
    );
  }

  /**
   * Get paginated synthesis history
   */
  getSyntheses(limit: number = 50, offset: number = 0): Observable<SynthesisListResponse> {
    const params = new HttpParams()
      .set('limit', limit.toString())
      .set('offset', offset.toString());

    return this.http.get<SynthesisListResponse>(
      `${this.apiUrl}/syntheses`,
      { params }
    );
  }

  /**
   * Get specific synthesis by ID
   */
  getSynthesis(id: string): Observable<Synthesis> {
    return this.http.get<Synthesis>(
      `${this.apiUrl}/syntheses/${id}`
    );
  }

  /**
   * Delete a synthesis
   */
  deleteSynthesis(id: string): Observable<void> {
    return this.http.delete<void>(
      `${this.apiUrl}/syntheses/${id}`
    );
  }

  /**
   * Get audio URL for a synthesis
   */
  getSynthesisAudioUrl(id: string): string {
    return `${this.apiUrl}/syntheses/${id}/audio`;
  }

  /**
   * Get audio download URL for a synthesis
   */
  getSynthesisAudioDownloadUrl(id: string): string {
    return `${this.apiUrl}/syntheses/${id}/audio?download=true`;
  }

  // ========== Voice Reference Endpoints ==========

  /**
   * Upload a voice reference for voice cloning
   */
  uploadVoiceReference(file: File, name: string): Observable<VoiceReferenceUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);

    return this.http.post<VoiceReferenceUploadResponse>(
      `${this.apiUrl}/voice-references`,
      formData
    );
  }

  /**
   * Get paginated voice references
   */
  getVoiceReferences(limit: number = 50, offset: number = 0): Observable<VoiceReferenceListResponse> {
    const params = new HttpParams()
      .set('limit', limit.toString())
      .set('offset', offset.toString());

    return this.http.get<VoiceReferenceListResponse>(
      `${this.apiUrl}/voice-references`,
      { params }
    );
  }

  /**
   * Get specific voice reference by ID
   */
  getVoiceReference(id: string): Observable<VoiceReference> {
    return this.http.get<VoiceReference>(
      `${this.apiUrl}/voice-references/${id}`
    );
  }

  /**
   * Delete a voice reference
   */
  deleteVoiceReference(id: string): Observable<void> {
    return this.http.delete<void>(
      `${this.apiUrl}/voice-references/${id}`
    );
  }

  /**
   * Get voice reference audio URL
   */
  getVoiceReferenceAudioUrl(id: string): string {
    return `${this.apiUrl}/voice-references/${id}/audio`;
  }

  // ========== Model Endpoints ==========

  /**
   * Get list of available TTS models
   */
  getAvailableModels(): Observable<ModelListResponse> {
    return this.http.get<ModelListResponse>(`${this.apiUrl}/models/available`);
  }

  /**
   * Get supported languages for multilingual model
   */
  getSupportedLanguages(): Observable<{ languages: string[]; count: number }> {
    return this.http.get<{ languages: string[]; count: number }>(
      `${this.apiUrl}/models/languages`
    );
  }

  /**
   * Get model status
   */
  getModelStatus(modelName: string): Observable<{ model_name: string; is_loaded: boolean }> {
    return this.http.get<{ model_name: string; is_loaded: boolean }>(
      `${this.apiUrl}/models/status/${modelName}`
    );
  }

  /**
   * Preload a model
   */
  preloadModel(modelName: string): Observable<{ model_name: string; is_loaded: boolean }> {
    return this.http.post<{ model_name: string; is_loaded: boolean }>(
      `${this.apiUrl}/models/load/${modelName}`,
      null
    );
  }

  /**
   * Get GPU info
   */
  getGpuInfo(): Observable<any> {
    return this.http.get(`${this.apiUrl}/models/gpu-info`);
  }

  // ========== Health Endpoints ==========

  /**
   * Check API health
   */
  healthCheck(): Observable<{ status: string; message: string }> {
    return this.http.get<{ status: string; message: string }>(
      `${this.apiUrl}/health`
    );
  }

  /**
   * Get system information
   */
  getSystemInfo(): Observable<any> {
    return this.http.get(`${this.apiUrl}/info`);
  }
}
