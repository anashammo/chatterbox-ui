/**
 * Popup configuration interface
 */
export interface PopupConfig {
  type: PopupType;
  title: string;
  message: string;
  confirmText: string;
  cancelText?: string;
  variant?: 'default' | 'success' | 'error';
}

/**
 * Popup type enum
 */
export enum PopupType {
  ALERT = 'alert',
  CONFIRM = 'confirm'
}

/**
 * Popup result interface
 */
export interface PopupResult {
  confirmed: boolean;
}
