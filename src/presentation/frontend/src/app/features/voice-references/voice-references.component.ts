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
  isUploading: boolean = false;

  // Audio player
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
      this.selectedFile = input.files[0];
      // Auto-fill name from filename if empty
      if (!this.voiceName) {
        this.voiceName = this.selectedFile.name.replace(/\.[^/.]+$/, '');
      }
    }
  }

  uploadVoiceReference(): void {
    if (!this.selectedFile || !this.voiceName.trim()) {
      this.error = 'Please select a file and enter a name';
      return;
    }

    this.isUploading = true;
    this.error = null;
    this.successMessage = null;

    this.synthesisService.uploadVoiceReference(this.selectedFile, this.voiceName.trim())
      .subscribe({
        next: (response) => {
          this.successMessage = `Voice reference "${response.name}" uploaded successfully`;
          this.selectedFile = null;
          this.voiceName = '';
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
    return new Date(dateString).toLocaleDateString();
  }
}
