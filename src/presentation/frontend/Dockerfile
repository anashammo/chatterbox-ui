FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies with BuildKit cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.npm \
    npm install

# Install Angular CLI globally with cache mount
RUN --mount=type=cache,target=/root/.npm \
    npm install -g @angular/cli

# Copy application code
COPY . .

# Install wget for health check (not included in node:18-alpine by default)
RUN apk add --no-cache wget

# Expose Angular dev server port
EXPOSE 4201

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:4201/ || exit 1

# Start Angular dev server
CMD ["ng", "serve", "--host", "0.0.0.0", "--port", "4201", "--poll", "1000"]
