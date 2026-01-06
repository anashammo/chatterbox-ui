import { Component, OnInit, OnDestroy } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { SynthesisService } from '../../core/services/synthesis.service';
import { ApiService } from '../../core/services/api.service';
import { Synthesis } from '../../core/models/synthesis.model';

@Component({
  selector: 'app-synthesis-detail',
  templateUrl: './synthesis-detail.component.html',
  styleUrls: ['./synthesis-detail.component.css']
})
export class SynthesisDetailComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  synthesis: Synthesis | null = null;
  isLoading: boolean = true;
  error: string | null = null;

  // Audio player
  audioElement: HTMLAudioElement | null = null;
  isPlaying: boolean = false;

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private synthesisService: SynthesisService,
    private apiService: ApiService
  ) {}

  ngOnInit(): void {
    const id = this.route.snapshot.paramMap.get('id');
    if (id) {
      this.loadSynthesis(id);
    }
  }

  ngOnDestroy(): void {
    this.stopAudio();
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadSynthesis(id: string): void {
    this.isLoading = true;
    this.synthesisService.loadSynthesis(id).subscribe({
      next: (synthesis) => {
        this.synthesis = synthesis;
        this.isLoading = false;
      },
      error: (err) => {
        this.error = err.error?.detail || err.message || 'Failed to load synthesis';
        this.isLoading = false;
      }
    });
  }

  playAudio(): void {
    if (!this.synthesis) return;

    if (this.isPlaying) {
      this.stopAudio();
      return;
    }

    const audioUrl = this.apiService.getSynthesisAudioUrl(this.synthesis.id);
    this.audioElement = new Audio(audioUrl);
    this.audioElement.onended = () => {
      this.isPlaying = false;
    };
    this.audioElement.play().then(() => {
      this.isPlaying = true;
    }).catch(err => {
      console.error('Failed to play audio:', err);
      this.error = 'Failed to play audio';
    });
  }

  stopAudio(): void {
    if (this.audioElement) {
      this.audioElement.pause();
      this.audioElement = null;
    }
    this.isPlaying = false;
  }

  downloadAudio(): void {
    if (!this.synthesis) return;
    const downloadUrl = this.apiService.getSynthesisAudioDownloadUrl(this.synthesis.id);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.click();
  }

  deleteSynthesis(): void {
    if (!this.synthesis) return;
    if (confirm('Are you sure you want to delete this synthesis?')) {
      this.synthesisService.deleteSynthesis(this.synthesis.id).subscribe({
        next: () => {
          this.router.navigate(['/history']);
        },
        error: (err) => {
          this.error = err.message || 'Failed to delete synthesis';
        }
      });
    }
  }

  copyText(): void {
    if (!this.synthesis) return;
    navigator.clipboard.writeText(this.synthesis.input_text).then(() => {
      // Could show a toast notification here
    });
  }

  getStatusClass(status: string): string {
    switch (status) {
      case 'completed': return 'status-completed';
      case 'processing': return 'status-processing';
      case 'failed': return 'status-failed';
      default: return 'status-pending';
    }
  }

  formatDate(dateString: string): string {
    return new Date(dateString).toLocaleString();
  }

  formatDuration(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  getWordCount(text: string): number {
    if (!text) return 0;
    return text.trim().split(/\s+/).length;
  }

  goBack(): void {
    this.router.navigate(['/history']);
  }

  getModelIcon(model: string): string {
    switch (model?.toLowerCase()) {
      case 'turbo': return '⚡';
      case 'standard': return '🎯';
      case 'multilingual': return '🌍';
      default: return '🔊';
    }
  }

  isArabic(text: string): boolean {
    if (!text) return false;
    // Arabic Unicode range: \u0600-\u06FF (Arabic), \u0750-\u077F (Arabic Supplement)
    const arabicPattern = /[\u0600-\u06FF\u0750-\u077F]/;
    return arabicPattern.test(text);
  }
}
