# Frontend Environment Configuration

## Overview

The Angular frontend uses **TypeScript environment files** for configuration, not traditional `.env` files.

## Files

- **`environment.ts`** - Development configuration (used by `ng serve`)
- **`environment.prod.ts`** - Production configuration (used by `ng build --prod`)

## Configuration Variables

### Development (`environment.ts`)
```typescript
export const environment = {
  production: false,
  apiUrl: 'http://localhost:8001/api/v1'
};
```

### Production (`environment.prod.ts`)
```typescript
export const environment = {
  production: true,
  apiUrl: 'http://localhost:8001/api/v1'  // Update for production deployment
};
```

## Usage in Components/Services

Import the environment file:

```typescript
import { environment } from '../environments/environment';

// Use the configuration
const apiUrl = environment.apiUrl;
const isProduction = environment.production;
```

## Docker Configuration

When running in Docker, the frontend connects to the backend using:
- **Internal** (container-to-container): `http://backend:8001/api/v1`
- **External** (browser-to-backend): `http://localhost:8001/api/v1`

The browser makes requests from the user's machine, so it uses the external URL (`localhost:8001`).

## Changing the API URL

1. **Local Development**: Edit `environment.ts`
2. **Production Build**: Edit `environment.prod.ts`
3. **Docker**: No changes needed (configured correctly)

After changes:
```bash
# Rebuild the Angular app
ng build

# Or restart the dev server
ng serve
```

## Why TypeScript Instead of .env?

Angular's CLI uses TypeScript environment files because:
1. **Type safety** - Catch configuration errors at compile time
2. **Angular AOT** - Ahead-of-time compilation bakes config into the bundle
3. **Tree shaking** - Unused environment variables are removed from production builds
4. **IDE support** - IntelliSense and auto-completion work out of the box

## Adding New Environment Variables

To add a new configuration variable:

1. Add to `environment.ts`:
```typescript
export const environment = {
  production: false,
  apiUrl: 'http://localhost:8001/api/v1',
  newVariable: 'value'  // Add here
};
```

2. Add the same to `environment.prod.ts` with production value:
```typescript
export const environment = {
  production: true,
  apiUrl: 'https://api.yourdomain.com/api/v1',
  newVariable: 'production-value'  // Add here
};
```

3. Use in code:
```typescript
import { environment } from '../environments/environment';
console.log(environment.newVariable);
```

## Backend Environment Configuration

Backend uses traditional `.env` files located at:
- **Local Dev**: `src/presentation/api/.env.example`
- **Docker**: `src/presentation/api/.env.docker`

See backend documentation for details.
