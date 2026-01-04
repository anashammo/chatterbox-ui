import { Component, OnInit, OnDestroy } from '@angular/core';
import { Router } from '@angular/router';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { SynthesisService } from '../../core/services/synthesis.service';
import { ApiService } from '../../core/services/api.service';
import { Synthesis } from '../../core/models/synthesis.model';

@Component({
  selector: 'app-synthesis-history',
  templateUrl: './synthesis-history.component.html',
  styleUrls: ['./synthesis-history.component.css']
})
export class SynthesisHistoryComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  syntheses: Synthesis[] = [];
  isLoading: boolean = false;
  error: string | null = null;

  // Audio player state
  currentlyPlayingId: string | null = null;
  audioElement: HTMLAudioElement | null = null;

  constructor(
    private synthesisService: SynthesisService,
    private apiService: ApiService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.loadSyntheses();

    this.synthesisService.syntheses$
      .pipe(takeUntil(this.destroy$))
      .subscribe(syntheses => this.syntheses = syntheses);

    this.synthesisService.loading$
      .pipe(takeUntil(this.destroy$))
      .subscribe(loading => this.isLoading = loading);
  }

  ngOnDestroy(): void {
    this.stopAudio();
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadSyntheses(): void {
    this.synthesisService.loadSyntheses();
  }

  viewSynthesis(synthesis: Synthesis): void {
    this.router.navigate(['/synthesis', synthesis.id]);
  }

  deleteSynthesis(synthesis: Synthesis, event: Event): void {
    event.stopPropagation();
    if (confirm('Are you sure you want to delete this synthesis?')) {
      this.synthesisService.deleteSynthesis(synthesis.id).subscribe();
    }
  }

  playAudio(synthesis: Synthesis, event: Event): void {
    event.stopPropagation();

    if (this.currentlyPlayingId === synthesis.id) {
      this.stopAudio();
      return;
    }

    this.stopAudio();
    this.currentlyPlayingId = synthesis.id;

    const audioUrl = this.apiService.getSynthesisAudioUrl(synthesis.id);
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

  downloadAudio(synthesis: Synthesis, event: Event): void {
    event.stopPropagation();
    const downloadUrl = this.apiService.getSynthesisAudioDownloadUrl(synthesis.id);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.click();
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

  truncateText(text: string, maxLength: number = 100): string {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  }
}
