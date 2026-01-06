import { Component, OnInit, OnDestroy } from '@angular/core';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { SynthesisService } from '../../core/services/synthesis.service';
import { ApiService } from '../../core/services/api.service';
import { VoiceReference } from '../../core/models/voice-reference.model';

@Component({
  selector: 'app-voice-references',
  templateUrl: './voice-references.component.html',
  styleUrls: ['./voice-references.component.css']
})
export class VoiceReferencesComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  voiceReferences: VoiceReference[] = [];
  isLoading: boolean = false;
  error: string | null = null;
  successMessage: string | null = null;

  // Upload form
  selectedFile: File | null = null;
  voiceName: string = '';
  selectedLanguage: string = '';
  isUploading: boolean = false;

  // Available languages for voice references
  availableLanguages = [
    { code: '', name: 'Not specified' },
    { code: 'en', name: 'English' },
    { code: 'es', name: 'Spanish' },
    { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' },
    { code: 'it', name: 'Italian' },
    { code: 'pt', name: 'Portuguese' },
    { code: 'ar', name: 'Arabic' },
    { code: 'zh', name: 'Chinese' },
    { code: 'ja', name: 'Japanese' },
    { code: 'ko', name: 'Korean' },
    { code: 'ru', name: 'Russian' },
    { code: 'hi', name: 'Hindi' },
  ];

  // Preview audio player
  isPreviewPlaying: boolean = false;
  previewAudioElement: HTMLAudioElement | null = null;
  previewAudioUrl: string | null = null;

  // Audio player for voice references
  currentlyPlayingId: string | null = null;
  audioElement: HTMLAudioElement | null = null;

  constructor(
    private synthesisService: SynthesisService,
    private apiService: ApiService
  ) {}

  ngOnInit(): void {
    this.loadVoiceReferences();

    this.synthesisService.voiceReferences$
      .pipe(takeUntil(this.destroy$))
      .subscribe(refs => this.voiceReferences = refs);
  }

  ngOnDestroy(): void {
    this.stopAudio();
    this.cleanupPreviewAudio();
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadVoiceReferences(): void {
    this.isLoading = true;
    this.synthesisService.loadVoiceReferences();
    setTimeout(() => this.isLoading = false, 500);
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      // Stop and clean up previous preview
      this.cleanupPreviewAudio();

      this.selectedFile = input.files[0];
      // Auto-fill name from filename if empty
      if (!this.voiceName) {
        this.voiceName = this.selectedFile.name.replace(/\.[^/.]+$/, '');
      }

      // Create object URL for preview
      this.previewAudioUrl = URL.createObjectURL(this.selectedFile);
    }
  }

  removeSelectedFile(): void {
    this.cleanupPreviewAudio();
    this.selectedFile = null;
    this.selectedLanguage = '';
    // Clear file input
    const fileInput = document.getElementById('fileInput') as HTMLInputElement;
    if (fileInput) fileInput.value = '';
  }

  togglePreviewAudio(): void {
    if (this.isPreviewPlaying) {
      this.stopPreviewAudio();
    } else {
      this.playPreviewAudio();
    }
  }

  playPreviewAudio(): void {
    if (!this.previewAudioUrl) return;

    // Stop any playing voice reference audio
    this.stopAudio();

    this.previewAudioElement = new Audio(this.previewAudioUrl);
    this.previewAudioElement.onended = () => {
      this.isPreviewPlaying = false;
    };
    this.previewAudioElement.play().then(() => {
      this.isPreviewPlaying = true;
    }).catch(err => {
      console.error('Failed to play preview audio:', err);
      this.error = 'Failed to play preview audio';
      this.isPreviewPlaying = false;
    });
  }

  stopPreviewAudio(): void {
    if (this.previewAudioElement) {
      this.previewAudioElement.pause();
      this.previewAudioElement = null;
    }
    this.isPreviewPlaying = false;
  }

  cleanupPreviewAudio(): void {
    this.stopPreviewAudio();
    if (this.previewAudioUrl) {
      URL.revokeObjectURL(this.previewAudioUrl);
      this.previewAudioUrl = null;
    }
  }

  uploadVoiceReference(): void {
    if (!this.selectedFile || !this.voiceName.trim()) {
      this.error = 'Please select a file and enter a name';
      return;
    }

    // Stop and clean up preview audio before upload
    this.cleanupPreviewAudio();

    this.isUploading = true;
    this.error = null;
    this.successMessage = null;

    const language = this.selectedLanguage || undefined;
    this.synthesisService.uploadVoiceReference(this.selectedFile, this.voiceName.trim(), language)
      .subscribe({
        next: (response) => {
          this.successMessage = `Voice reference "${response.name}" uploaded successfully`;
          this.selectedFile = null;
          this.voiceName = '';
          this.selectedLanguage = '';
          this.isUploading = false;
          // Clear file input
          const fileInput = document.getElementById('fileInput') as HTMLInputElement;
          if (fileInput) fileInput.value = '';
        },
        error: (err) => {
          this.error = err.error?.detail || err.message || 'Upload failed';
          this.isUploading = false;
        }
      });
  }

  deleteVoiceReference(ref: VoiceReference): void {
    if (confirm(`Are you sure you want to delete "${ref.name}"?`)) {
      this.synthesisService.deleteVoiceReference(ref.id).subscribe({
        next: () => {
          this.successMessage = `Voice reference "${ref.name}" deleted`;
        },
        error: (err) => {
          this.error = err.message || 'Failed to delete voice reference';
        }
      });
    }
  }

  playAudio(ref: VoiceReference): void {
    if (this.currentlyPlayingId === ref.id) {
      this.stopAudio();
      return;
    }

    // Stop preview audio if playing
    this.stopPreviewAudio();

    this.stopAudio();
    this.currentlyPlayingId = ref.id;

    const audioUrl = this.apiService.getVoiceReferenceAudioUrl(ref.id);
    this.audioElement = new Audio(audioUrl);
    this.audioElement.onended = () => {
      this.currentlyPlayingId = null;
    };
    this.audioElement.play().catch(err => {
      console.error('Failed to play audio:', err);
      this.currentlyPlayingId = null;
    });
  }

  stopAudio(): void {
    if (this.audioElement) {
      this.audioElement.pause();
      this.audioElement = null;
    }
    this.currentlyPlayingId = null;
  }

  formatDate(dateString: string): string {
    return new Date(dateString).toLocaleString();
  }
}
