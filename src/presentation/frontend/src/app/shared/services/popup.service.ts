import { Injectable, ApplicationRef, ComponentRef, createComponent, EnvironmentInjector } from '@angular/core';
import { Subject, Observable } from 'rxjs';
import { PopupComponent } from '../components/popup/popup.component';
import { PopupConfig, PopupType, PopupResult } from '../models/popup.model';

/**
 * Service for displaying custom popup dialogs
 * Replaces browser native confirm() and alert() with styled popups
 */
@Injectable({
  providedIn: 'root'
})
export class PopupService {
  private popupComponentRef: ComponentRef<PopupComponent> | null = null;

  constructor(
    private appRef: ApplicationRef,
    private injector: EnvironmentInjector
  ) {}

  /**
   * Show confirmation dialog with Cancel/Confirm buttons
   * @param title - Dialog title
   * @param message - Dialog message
   * @param confirmText - Confirm button text (default: 'Confirm')
   * @param cancelText - Cancel button text (default: 'Cancel')
   * @returns Observable<boolean> - true if confirmed, false if cancelled
   */
  confirm(title: string, message: string, confirmText = 'Confirm', cancelText = 'Cancel'): Observable<boolean> {
    const config: PopupConfig = {
      type: PopupType.CONFIRM,
      title,
      message,
      confirmText,
      cancelText
    };
    return this.show(config);
  }

  /**
   * Show alert/notification with OK button
   * @param title - Dialog title
   * @param message - Dialog message
   * @param okText - OK button text (default: 'OK')
   * @returns Observable<boolean> - true when dismissed
   */
  alert(title: string, message: string, okText = 'OK'): Observable<boolean> {
    const config: PopupConfig = {
      type: PopupType.ALERT,
      title,
      message,
      confirmText: okText
    };
    return this.show(config);
  }

  /**
   * Show success notification (green theme variant)
   * @param message - Success message
   * @param title - Dialog title (default: 'Success')
   * @returns Observable<boolean> - true when dismissed
   */
  success(message: string, title = 'Success'): Observable<boolean> {
    const config: PopupConfig = {
      type: PopupType.ALERT,
      variant: 'success',
      title,
      message,
      confirmText: 'OK'
    };
    return this.show(config);
  }

  /**
   * Show error notification (red theme variant)
   * @param message - Error message
   * @param title - Dialog title (default: 'Error')
   * @returns Observable<boolean> - true when dismissed
   */
  error(message: string, title = 'Error'): Observable<boolean> {
    const config: PopupConfig = {
      type: PopupType.ALERT,
      variant: 'error',
      title,
      message,
      confirmText: 'OK'
    };
    return this.show(config);
  }

  /**
   * Core method to display popup
   * Creates popup component dynamically and appends to document body
   * @param config - Popup configuration
   * @returns Observable<boolean> - Result from user interaction
   */
  private show(config: PopupConfig): Observable<boolean> {
    // Remove existing popup if any
    this.close();

    // Create result subject
    const result$ = new Subject<boolean>();

    // Dynamically create popup component
    this.popupComponentRef = createComponent(PopupComponent, {
      environmentInjector: this.injector
    });

    // Set config and result handler
    this.popupComponentRef.instance.config = config;
    this.popupComponentRef.instance.result.subscribe((result: PopupResult) => {
      result$.next(result.confirmed);
      result$.complete();
      this.close();
    });

    // Attach to DOM
    this.appRef.attachView(this.popupComponentRef.hostView);
    const domElement = this.popupComponentRef.location.nativeElement;
    document.body.appendChild(domElement);

    return result$.asObservable();
  }

  /**
   * Close and cleanup popup
   * Removes component from DOM and destroys component reference
   */
  private close(): void {
    if (this.popupComponentRef) {
      this.appRef.detachView(this.popupComponentRef.hostView);
      this.popupComponentRef.destroy();
      this.popupComponentRef = null;
    }
  }
}
