import { Component, OnInit, OnDestroy } from '@angular/core';
import { Router } from '@angular/router';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { SynthesisService } from '../../core/services/synthesis.service';
import { ApiService } from '../../core/services/api.service';
import { TTSModel, SynthesisCreateRequest } from '../../core/models/synthesis.model';
import { VoiceReference } from '../../core/models/voice-reference.model';

@Component({
  selector: 'app-synthesize',
  templateUrl: './synthesize.component.html',
  styleUrls: ['./synthesize.component.css']
})
export class SynthesizeComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  // Form state
  text: string = '';
  selectedModel: string = 'multilingual';
  selectedVoiceReferenceId: string = '';
  selectedLanguage: string = '';
  cfgWeight: number = 0.5;
  exaggeration: number = 0.5;

  // Data
  models: TTSModel[] = [];
  voiceReferences: VoiceReference[] = [];
  supportedLanguages: string[] = [];

  // UI state
  isLoading: boolean = false;
  error: string | null = null;
  showAdvanced: boolean = false;

  // Character limit
  readonly maxTextLength = 5000;

  constructor(
    private synthesisService: SynthesisService,
    private apiService: ApiService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.loadModels();
    this.loadVoiceReferences();
    this.loadLanguages();

    // Subscribe to loading state
    this.synthesisService.loading$
      .pipe(takeUntil(this.destroy$))
      .subscribe(loading => this.isLoading = loading);

    // Subscribe to error state
    this.synthesisService.error$
      .pipe(takeUntil(this.destroy$))
      .subscribe(error => this.error = error);
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadModels(): void {
    this.apiService.getAvailableModels().subscribe({
      next: (response) => {
        this.models = response.models;
      },
      error: (err) => console.error('Failed to load models:', err)
    });
  }

  loadVoiceReferences(): void {
    this.synthesisService.loadVoiceReferences();
    this.synthesisService.voiceReferences$
      .pipe(takeUntil(this.destroy$))
      .subscribe(refs => this.voiceReferences = refs);
  }

  loadLanguages(): void {
    this.apiService.getSupportedLanguages().subscribe({
      next: (response) => {
        this.supportedLanguages = response.languages;
      },
      error: (err) => console.error('Failed to load languages:', err)
    });
  }

  get isMultilingual(): boolean {
    return this.selectedModel === 'multilingual';
  }

  get textLength(): number {
    return this.text.length;
  }

  get isValid(): boolean {
    if (!this.text.trim()) return false;
    if (this.text.length > this.maxTextLength) return false;
    if (this.isMultilingual && !this.selectedLanguage) return false;
    return true;
  }

  synthesize(): void {
    if (!this.isValid) return;

    this.error = null;

    const request: SynthesisCreateRequest = {
      text: this.text.trim(),
      model: this.selectedModel,
      cfg_weight: this.cfgWeight,
      exaggeration: this.exaggeration
    };

    if (this.selectedVoiceReferenceId) {
      request.voice_reference_id = this.selectedVoiceReferenceId;
    }

    if (this.isMultilingual && this.selectedLanguage) {
      request.language = this.selectedLanguage;
    }

    this.synthesisService.createSynthesis(request).subscribe({
      next: (synthesis) => {
        this.router.navigate(['/synthesis', synthesis.id]);
      },
      error: (err) => {
        this.error = err.error?.detail || err.message || 'Synthesis failed';
      }
    });
  }

  clearText(): void {
    this.text = '';
  }

  toggleAdvanced(): void {
    this.showAdvanced = !this.showAdvanced;
  }

  getModelDescription(model: TTSModel): string {
    return `${model.display_name} (${model.parameters}) - ${model.description}`;
  }

  get selectedModelDescription(): string {
    const model = this.models.find(m => m.name === this.selectedModel);
    return model?.description || '';
  }

  getLanguageDisplay(lang: string): string {
    return lang.toUpperCase();
  }
}
