# Whisper Transcription Frontend

Angular frontend application for the Whisper voice-to-text transcription system.

**Location**: `src/presentation/frontend/` (Part of the Presentation Layer)

## Features

- **Upload Component**: Drag & drop audio file upload with language selection
- **Transcription Detail**: Real-time transcription status with auto-polling
- **History View**: List of all transcriptions with filtering and search
- **Simple UI**: Clean, minimal interface focused on functionality

## Prerequisites

- Node.js 18 or higher
- npm 9 or higher
- Backend API running on http://localhost:8001

## Installation

1. **Navigate to Frontend Directory**

```bash
# From project root
cd src/presentation/frontend
```

2. **Install Dependencies**

```bash
npm install
```

3. **Configure API URL**

The API URL is configured in `src/environments/environment.ts`:

```typescript
export const environment = {
  production: false,
  apiUrl: 'http://localhost:8001/api/v1'
};
```

Update this if your backend API runs on a different URL.

## Running the Application

### Development Server

```bash
npm start
```

Navigate to `http://localhost:4200/`. The application will automatically reload if you change any source files.

### Production Build

```bash
npm run build
```

The build artifacts will be stored in the `dist/` directory.

## Application Structure

```
src/presentation/frontend/         # Frontend (Presentation Layer)
├── src/
│   ├── app/
│   │   ├── core/                  # Core services and models
│   │   │   ├── models/           # TypeScript interfaces
│   │   │   └── services/         # API and business logic services
│   │   ├── features/              # Feature modules
│   │   │   ├── upload/           # Upload component
│   │   │   ├── transcription/    # Transcription detail component
│   │   │   └── history/          # History list component
│   │   ├── app.component.*       # Root component
│   │   ├── app-routing.module.ts # Routing configuration
│   │   └── app.module.ts         # App module
│   ├── environments/              # Environment configurations
│   ├── styles.css                # Global styles
│   ├── index.html                # HTML entry point
│   └── main.ts                   # TypeScript entry point
├── angular.json                   # Angular CLI configuration
├── package.json                   # Dependencies
├── tsconfig.json                 # TypeScript configuration
└── README.md                     # This file
```

## Usage

### 1. Upload Audio

- Navigate to the home page (`/`)
- Drag and drop an audio file or click to browse
- Optionally select a language
- Click "Start Transcription"

### 2. View Transcription

- After upload, you'll be redirected to the transcription detail page
- The page automatically polls for updates while processing
- Once complete, you can copy or download the transcription

### 3. View History

- Click "View History" from any page
- See all past transcriptions
- Click on any transcription to view details

## API Integration

The frontend communicates with the backend API through:

- **ApiService**: HTTP client for API calls
- **TranscriptionService**: Business logic and state management

### Key Services

**ApiService** (`core/services/api.service.ts`):
- `uploadAudio(file, language)` - Upload and transcribe
- `getTranscriptions(limit, offset)` - Get history
- `getTranscription(id)` - Get single transcription
- `healthCheck()` - Check API health

**TranscriptionService** (`core/services/transcription.service.ts`):
- Manages application state with RxJS BehaviorSubjects
- Handles polling for processing transcriptions
- Provides helper methods for formatting

## Supported Audio Formats

- MP3 (audio/mpeg)
- WAV (audio/wav)
- M4A (audio/mp4, audio/x-m4a)
- FLAC (audio/flac)
- OGG (audio/ogg)
- WEBM (audio/webm)

Maximum file size: 25MB

## Styling

The application uses a simple, clean design with:
- Custom CSS (no external frameworks)
- Responsive layout
- Consistent color scheme
- Loading states and animations

## Troubleshooting

### API Connection Issues

If you see "Failed to connect to API":
1. Ensure backend is running on http://localhost:8001
2. Check CORS settings in backend allow http://localhost:4200
3. Verify `environment.ts` has correct API URL

### Build Errors

If you encounter build errors:
1. Delete `node_modules` and `package-lock.json`
2. Run `npm install` again
3. Clear Angular cache: `npm run ng cache clean`

### TypeScript Errors

If you see TypeScript errors:
1. Ensure TypeScript version matches `package.json`
2. Run `npm install` to update dependencies
3. Restart your IDE/editor

## Development

### Adding New Features

1. Create new component: `ng generate component features/my-feature`
2. Add route in `app-routing.module.ts`
3. Update navigation links in components

### Code Style

- Use TypeScript strict mode
- Follow Angular style guide
- Use RxJS observables for async operations
- Implement OnDestroy for cleanup

## License

This project is for educational and internal use.
