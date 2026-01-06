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

  // Pagination state
  currentPage: number = 1;
  pageSize: number = 5;
  totalItems: number = 0;

  // Audio player state
  currentlyPlayingId: string | null = null;
  audioElement: HTMLAudioElement | null = null;

  constructor(
    private synthesisService: SynthesisService,
    private apiService: ApiService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.loadPage(1);

    this.synthesisService.syntheses$
      .pipe(takeUntil(this.destroy$))
      .subscribe(syntheses => this.syntheses = syntheses);

    this.synthesisService.totalSyntheses$
      .pipe(takeUntil(this.destroy$))
      .subscribe(total => this.totalItems = total);

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
    const offset = (this.currentPage - 1) * this.pageSize;
    this.synthesisService.loadSyntheses(this.pageSize, offset);
  }

  loadHistory(): void {
    this.error = null;
    this.loadSyntheses();
  }

  // Pagination methods
  loadPage(page: number): void {
    this.currentPage = page;
    this.loadSyntheses();
  }

  get totalPages(): number {
    return Math.ceil(this.totalItems / this.pageSize);
  }

  get canGoPrevious(): boolean {
    return this.currentPage > 1;
  }

  get canGoNext(): boolean {
    return this.currentPage < this.totalPages;
  }

  previousPage(): void {
    if (this.canGoPrevious) {
      this.loadPage(this.currentPage - 1);
    }
  }

  nextPage(): void {
    if (this.canGoNext) {
      this.loadPage(this.currentPage + 1);
    }
  }

  firstPage(): void {
    if (this.currentPage !== 1) {
      this.loadPage(1);
    }
  }

  lastPage(): void {
    if (this.currentPage !== this.totalPages) {
      this.loadPage(this.totalPages);
    }
  }

  formatDuration(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
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
