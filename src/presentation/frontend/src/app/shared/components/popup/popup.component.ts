import { Component, Output, EventEmitter, HostListener, OnInit, OnDestroy } from '@angular/core';
import { trigger, state, style, transition, animate } from '@angular/animations';
import { PopupConfig, PopupType, PopupResult } from '../../models/popup.model';

/**
 * Custom popup/modal component
 * Replaces browser native confirm() and alert() dialogs
 */
@Component({
  selector: 'app-popup',
  templateUrl: './popup.component.html',
  styleUrls: ['./popup.component.css'],
  animations: [
    // Backdrop fade animation
    trigger('backdropAnimation', [
      transition(':enter', [
        style({ opacity: 0 }),
        animate('200ms ease-out', style({ opacity: 1 }))
      ]),
      transition(':leave', [
        animate('200ms ease-in', style({ opacity: 0 }))
      ])
    ]),
    // Modal slide and fade animation
    trigger('modalAnimation', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(-20px) scale(0.95)' }),
        animate('250ms ease-out', style({ opacity: 1, transform: 'translateY(0) scale(1)' }))
      ]),
      transition(':leave', [
        animate('200ms ease-in', style({ opacity: 0, transform: 'translateY(-10px) scale(0.98)' }))
      ])
    ])
  ]
})
export class PopupComponent implements OnInit, OnDestroy {
  @Output() result = new EventEmitter<PopupResult>();

  config!: PopupConfig;
  PopupType = PopupType;
  isVisible = false;

  ngOnInit(): void {
    // Trigger animation after component is created
    setTimeout(() => {
      this.isVisible = true;
    }, 10);

    // Prevent body scroll when popup is open
    document.body.style.overflow = 'hidden';
  }

  ngOnDestroy(): void {
    // Restore body scroll
    document.body.style.overflow = '';
  }

  /**
   * Handle ESC key to cancel/close
   */
  @HostListener('document:keydown.escape')
  onEscapeKey(): void {
    if (this.config.type === PopupType.CONFIRM) {
      this.onCancel();
    } else {
      this.onConfirm();
    }
  }

  /**
   * Handle Enter key to confirm
   */
  @HostListener('document:keydown.enter')
  onEnterKey(): void {
    this.onConfirm();
  }

  /**
   * Handle backdrop click to cancel/close
   */
  onBackdropClick(): void {
    if (this.config.type === PopupType.CONFIRM) {
      this.onCancel();
    } else {
      this.onConfirm();
    }
  }

  /**
   * Prevent modal click from closing popup
   */
  onModalClick(event: Event): void {
    event.stopPropagation();
  }

  /**
   * Handle confirm button click
   */
  onConfirm(): void {
    this.isVisible = false;
    setTimeout(() => {
      this.result.emit({ confirmed: true });
    }, 200); // Wait for exit animation
  }

  /**
   * Handle cancel button click
   */
  onCancel(): void {
    this.isVisible = false;
    setTimeout(() => {
      this.result.emit({ confirmed: false });
    }, 200); // Wait for exit animation
  }

  /**
   * Get CSS class for variant styling
   */
  getVariantClass(): string {
    return this.config.variant ? `popup-${this.config.variant}` : 'popup-default';
  }
}
