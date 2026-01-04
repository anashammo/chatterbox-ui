# Chatterbox TTS Frontend

Angular frontend application for the Chatterbox Text-to-Speech synthesis system.

**Location**: `src/presentation/frontend/` (Part of the Presentation Layer)

## Features

- **Text Input**: Text area with character count and validation for TTS synthesis
- **Voice Selection**: Choose from uploaded voice references for voice cloning
- **Model Selection**: Select TTS model (turbo, standard, multilingual)
- **Parameter Tuning**: CFG weight and exaggeration sliders for voice control
- **Audio Playback**: Play and download synthesized audio
- **History View**: List of all syntheses with status indicators
- **Voice Library**: Manage voice reference audio files

## Prerequisites

- Node.js 18 or higher
- npm 9 or higher
- Backend API running on http://localhost:8002

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
  apiUrl: 'http://localhost:8002/api/v1'
};
```

Update this if your backend API runs on a different URL.

## Running the Application

### Development Server

```bash
npm start
```

Or with specific port:

```bash
ng serve --port 4201
```

Navigate to `http://localhost:4201/`. The application will automatically reload if you change any source files.

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
│   │   │   ├── models/           # TypeScript interfaces (Synthesis, VoiceReference)
│   │   │   └── services/         # API and state management services
│   │   ├── features/              # Feature modules
│   │   │   ├── synthesis/        # TTS synthesis component
│   │   │   ├── voice-library/    # Voice reference management
│   │   │   └── history/          # Synthesis history component
│   │   ├── shared/               # Shared components (header, footer, audio player)
│   │   ├── app.component.*       # Root component
│   │   ├── app.routes.ts         # Routing configuration
│   │   └── app.config.ts         # App configuration
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

### 1. Text-to-Speech Synthesis

- Navigate to the synthesis page
- Enter text to synthesize (up to 5000 characters)
- Select a voice reference (optional, for voice cloning)
- Choose a TTS model (turbo, standard, multilingual)
- Adjust CFG weight and exaggeration parameters
- Click "Synthesize" to generate audio

### 2. Voice Library

- Upload voice reference audio (5-30 seconds recommended)
- Preview and manage uploaded voice references
- Delete unused voice references

### 3. View History

- See all past syntheses with status indicators
- Play or download generated audio
- Delete old syntheses

## API Integration

The frontend communicates with the backend API through:

- **ApiService**: HTTP client for API calls
- **SynthesisService**: State management for TTS operations

### Key Services

**ApiService** (`core/services/api.service.ts`):
- `createSynthesis(request)` - Create new TTS synthesis
- `getSyntheses(limit, offset)` - Get synthesis history
- `getSynthesis(id)` - Get single synthesis
- `getVoiceReferences()` - Get voice reference library
- `uploadVoiceReference(file, name)` - Upload voice reference
- `deleteVoiceReference(id)` - Delete voice reference
- `getAvailableModels()` - Get available TTS models

**SynthesisService** (`core/services/synthesis.service.ts`):
- Manages application state with RxJS BehaviorSubjects
- Provides reactive streams for syntheses, voice references, and models

## Supported Audio Formats

For voice reference uploads:
- WAV (audio/wav)
- MP3 (audio/mpeg)
- FLAC (audio/flac)
- OGG (audio/ogg)

Maximum file size: 10MB
Recommended duration: ~10 seconds

## Styling

The application uses:
- Custom CSS with dark mode theme
- Responsive layout
- Consistent color scheme
- Loading states and animations

## Troubleshooting

### API Connection Issues

If you see connection errors:
1. Ensure backend is running on http://localhost:8002
2. Check CORS settings in backend allow http://localhost:4201
3. Verify `environment.ts` has correct API URL

### Build Errors

If you encounter build errors:
1. Delete `node_modules` and `package-lock.json`
2. Run `npm install` again
3. Clear Angular cache: `npm run ng cache clean`

## Development

### Adding New Features

1. Create new component: `ng generate component features/my-feature`
2. Add route in `app.routes.ts`
3. Update navigation links in components

### Code Style

- Use TypeScript strict mode
- Follow Angular style guide
- Use RxJS observables for async operations
- Implement OnDestroy for cleanup

## License

This project is for educational and internal use.
