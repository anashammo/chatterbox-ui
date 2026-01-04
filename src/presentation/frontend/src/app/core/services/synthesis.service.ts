import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, Subject } from 'rxjs';
import { takeUntil, tap, catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';
import { Synthesis, SynthesisCreateRequest, TTSModel } from '../models/synthesis.model';
import { VoiceReference } from '../models/voice-reference.model';
import { ApiService } from './api.service';

/**
 * Service for managing TTS synthesis state and operations
 */
@Injectable({
  providedIn: 'root'
})
export class SynthesisService {
  // Current synthesis being viewed/edited
  private currentSynthesisSubject = new BehaviorSubject<Synthesis | null>(null);
  currentSynthesis$ = this.currentSynthesisSubject.asObservable();

  // All syntheses for history
  private synthesesSubject = new BehaviorSubject<Synthesis[]>([]);
  syntheses$ = this.synthesesSubject.asObservable();

  // Voice references for selection
  private voiceReferencesSubject = new BehaviorSubject<VoiceReference[]>([]);
  voiceReferences$ = this.voiceReferencesSubject.asObservable();

  // Available TTS models
  private modelsSubject = new BehaviorSubject<TTSModel[]>([]);
  models$ = this.modelsSubject.asObservable();

  // Loading state
  private loadingSubject = new BehaviorSubject<boolean>(false);
  loading$ = this.loadingSubject.asObservable();

  // Error state
  private errorSubject = new BehaviorSubject<string | null>(null);
  error$ = this.errorSubject.asObservable();

  constructor(private apiService: ApiService) {}

  /**
   * Load synthesis by ID
   * Uses tap() instead of subscribe() to avoid duplicate HTTP requests
   */
  loadSynthesis(id: string): Observable<Synthesis> {
    this.loadingSubject.next(true);
    this.errorSubject.next(null);

    return this.apiService.getSynthesis(id).pipe(
      tap((synthesis) => {
        this.currentSynthesisSubject.next(synthesis);
        this.loadingSubject.next(false);
      }),
      catchError((err) => {
        this.errorSubject.next(err.message || 'Failed to load synthesis');
        this.loadingSubject.next(false);
        return throwError(() => err);
      })
    );
  }

  /**
   * Load all syntheses for history
   */
  loadSyntheses(limit: number = 50, offset: number = 0): void {
    this.loadingSubject.next(true);
    this.errorSubject.next(null);

    this.apiService.getSyntheses(limit, offset).subscribe({
      next: (response) => {
        this.synthesesSubject.next(response.syntheses);
        this.loadingSubject.next(false);
      },
      error: (err) => {
        this.errorSubject.next(err.message || 'Failed to load syntheses');
        this.loadingSubject.next(false);
      }
    });
  }

  /**
   * Create a new synthesis
   * Uses tap() instead of subscribe() to avoid duplicate HTTP requests
   */
  createSynthesis(request: SynthesisCreateRequest): Observable<Synthesis> {
    this.loadingSubject.next(true);
    this.errorSubject.next(null);

    return this.apiService.createSynthesis(request).pipe(
      tap((synthesis) => {
        this.currentSynthesisSubject.next(synthesis);
        // Add to beginning of list
        const current = this.synthesesSubject.getValue();
        this.synthesesSubject.next([synthesis, ...current]);
        this.loadingSubject.next(false);
      }),
      catchError((err) => {
        this.errorSubject.next(err.error?.detail || err.message || 'Synthesis failed');
        this.loadingSubject.next(false);
        return throwError(() => err);
      })
    );
  }

  /**
   * Delete a synthesis
   * Uses tap() instead of subscribe() to avoid duplicate HTTP requests
   */
  deleteSynthesis(id: string): Observable<void> {
    return this.apiService.deleteSynthesis(id).pipe(
      tap(() => {
        // Remove from list
        const current = this.synthesesSubject.getValue();
        this.synthesesSubject.next(current.filter(s => s.id !== id));

        // Clear current if it was deleted
        if (this.currentSynthesisSubject.getValue()?.id === id) {
          this.currentSynthesisSubject.next(null);
        }
      }),
      catchError((err) => {
        // On 404, item doesn't exist - remove from local list anyway
        if (err.status === 404) {
          const current = this.synthesesSubject.getValue();
          this.synthesesSubject.next(current.filter(s => s.id !== id));
        }
        this.errorSubject.next(err.error?.detail || err.message || 'Failed to delete synthesis');
        return throwError(() => err);
      })
    );
  }

  /**
   * Load voice references for selection
   */
  loadVoiceReferences(): void {
    this.apiService.getVoiceReferences().subscribe({
      next: (response) => {
        this.voiceReferencesSubject.next(response.voice_references);
      },
      error: (err) => {
        console.error('Failed to load voice references:', err);
      }
    });
  }

  /**
   * Upload a voice reference
   * Uses tap() instead of subscribe() to avoid duplicate HTTP requests
   */
  uploadVoiceReference(file: File, name: string): Observable<any> {
    return this.apiService.uploadVoiceReference(file, name).pipe(
      tap((response) => {
        // Add the new voice reference to the list immediately
        const current = this.voiceReferencesSubject.getValue();
        const newRef: VoiceReference = {
          id: response.id,
          name: response.name,
          original_filename: response.original_filename,
          file_size_bytes: response.file_size_mb * 1024 * 1024,
          file_size_mb: response.file_size_mb,
          mime_type: 'audio/wav',
          duration_seconds: response.duration_seconds,
          created_at: new Date().toISOString()
        };
        this.voiceReferencesSubject.next([newRef, ...current]);
      }),
      catchError((err) => {
        this.errorSubject.next(err.error?.detail || err.message || 'Upload failed');
        return throwError(() => err);
      })
    );
  }

  /**
   * Delete a voice reference
   * Uses tap() instead of subscribe() to avoid duplicate HTTP requests
   */
  deleteVoiceReference(id: string): Observable<void> {
    return this.apiService.deleteVoiceReference(id).pipe(
      tap(() => {
        // Remove from list on success
        const current = this.voiceReferencesSubject.getValue();
        this.voiceReferencesSubject.next(current.filter(vr => vr.id !== id));
      }),
      catchError((err) => {
        // On 404, item doesn't exist - remove from local list anyway
        if (err.status === 404) {
          const current = this.voiceReferencesSubject.getValue();
          this.voiceReferencesSubject.next(current.filter(vr => vr.id !== id));
        }
        this.errorSubject.next(err.error?.detail || err.message || 'Failed to delete voice reference');
        return throwError(() => err);
      })
    );
  }

  /**
   * Load available TTS models
   */
  loadModels(): void {
    this.apiService.getAvailableModels().subscribe({
      next: (response) => {
        this.modelsSubject.next(response.models);
      },
      error: (err) => {
        console.error('Failed to load models:', err);
      }
    });
  }

  /**
   * Get audio URL for current synthesis
   */
  getAudioUrl(synthesisId: string): string {
    return this.apiService.getSynthesisAudioUrl(synthesisId);
  }

  /**
   * Get audio download URL
   */
  getAudioDownloadUrl(synthesisId: string): string {
    return this.apiService.getSynthesisAudioDownloadUrl(synthesisId);
  }

  /**
   * Clear current synthesis
   */
  clearCurrent(): void {
    this.currentSynthesisSubject.next(null);
  }

  /**
   * Clear error
   */
  clearError(): void {
    this.errorSubject.next(null);
  }
}
