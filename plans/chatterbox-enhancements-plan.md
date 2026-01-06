# Plan: Chatterbox TTS Enhancements

## Overview

Implement JIT warmup on startup, synthesis history improvements, and default model change.

---

## Issues to Address

| # | Issue | Description |
|---|-------|-------------|
| 1 | JIT Warmup | First request has ~20s delay due to JIT compilation |
| 2a | Text Trimming | History view truncates text to 100 chars |
| 2b | No Pagination | History loads 50 items with no page controls |
| 2c | No Voice Ref Info | Only UUID shown, not name/language |
| 2d | Arabic Display | "????????" shown instead of Arabic text |
| 3 | Default Model | Should be "multilingual" not "turbo" |

---

## Implementation Plan

### Phase 1: Fix Arabic Text Encoding (Prerequisite)

**Root Cause:** JSON response uses `ensure_ascii=True` by default, escaping non-ASCII.

**File:** `src/presentation/api/main.py`

Add custom UTF-8 JSON response class:
```python
from fastapi.responses import JSONResponse
import json

class UTF8JSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,  # Allow Unicode characters
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

app = FastAPI(
    # ... existing config ...
    default_response_class=UTF8JSONResponse,
)
```

---

### Phase 2: Default Model Change to "multilingual"

**Files to modify:**

| File | Change |
|------|--------|
| `src/infrastructure/config/settings.py` | `tts_default_model: str = "multilingual"` |
| `src/presentation/api/schemas/synthesis_schema.py` | `model: str = Field(default="multilingual", ...)` |
| `src/presentation/api/.env.example` | `TTS_DEFAULT_MODEL=multilingual` |
| `docker-compose.yml` | `TTS_DEFAULT_MODEL:-multilingual`, `PRELOAD_MODEL_LIST:-multilingual,turbo,standard` |
| `src/presentation/frontend/.../synthesize.component.ts` | `selectedModel: string = 'multilingual'` |

---

### Phase 3: Voice Reference Info in Synthesis Response

**Backend Changes:**

1. **Add schema** (`src/presentation/api/schemas/synthesis_schema.py`):
```python
class VoiceReferenceInfo(BaseModel):
    id: str
    name: str
    language: Optional[str] = None

class SynthesisResponse(BaseModel):
    # ... existing fields ...
    voice_reference: Optional[VoiceReferenceInfo] = None
```

2. **Add DTO** (`src/application/dto/synthesis_dto.py`):
```python
@dataclass
class VoiceReferenceInfoDTO:
    id: str
    name: str
    language: Optional[str] = None
```

3. **Eager load in repository** (`src/infrastructure/persistence/repositories/sqlite_synthesis_repository.py`):
```python
from sqlalchemy.orm import joinedload

def get_all(self, limit, offset):
    models = (
        self.db.query(SynthesisModel)
        .options(joinedload(SynthesisModel.voice_reference))
        .order_by(SynthesisModel.created_at.desc())
        .offset(offset).limit(limit).all()
    )
```

**Frontend Changes:**

1. **Update model** (`src/presentation/frontend/src/app/core/models/synthesis.model.ts`):
```typescript
export interface VoiceReferenceInfo {
  id: string;
  name: string;
  language: string | null;
}

export interface Synthesis {
  // ... existing ...
  voice_reference: VoiceReferenceInfo | null;
}
```

2. **Display in template** (`src/presentation/frontend/src/app/features/synthesis-history/synthesis-history.component.html`):
```html
<span class="meta-item" *ngIf="synthesis.voice_reference">
  Voice: {{ synthesis.voice_reference.name }}
  <span *ngIf="synthesis.voice_reference.language">({{ synthesis.voice_reference.language | uppercase }})</span>
</span>
```

---

### Phase 4: Pagination (10 items per page)

**Frontend Changes:**

1. **Component** (`src/presentation/frontend/src/app/features/synthesis-history/synthesis-history.component.ts`):
```typescript
currentPage: number = 1;
pageSize: number = 10;
totalItems: number = 0;
totalPages: number = 0;

loadSyntheses(page: number = 1): void {
  this.currentPage = page;
  const offset = (page - 1) * this.pageSize;
  this.apiService.getSyntheses(this.pageSize, offset).subscribe({
    next: (response) => {
      this.synthesesSubject.next(response.syntheses);
      this.totalItems = response.total;
      this.totalPages = Math.ceil(response.total / this.pageSize);
    }
  });
}

nextPage(): void { if (this.currentPage < this.totalPages) this.loadSyntheses(this.currentPage + 1); }
previousPage(): void { if (this.currentPage > 1) this.loadSyntheses(this.currentPage - 1); }
```

2. **Template** (`src/presentation/frontend/src/app/features/synthesis-history/synthesis-history.component.html`):
```html
<div class="pagination" *ngIf="totalPages > 1">
  <button [disabled]="currentPage === 1" (click)="previousPage()">Previous</button>
  <span>Page {{ currentPage }} of {{ totalPages }} ({{ totalItems }} total)</span>
  <button [disabled]="currentPage === totalPages" (click)="nextPage()">Next</button>
</div>
```

---

### Phase 5: Show Full Text (Remove Truncation)

**File:** `src/presentation/frontend/src/app/features/synthesis-history/synthesis-history.component.html`

Change:
```html
{{ truncateText(synthesis.input_text) }}
```
To:
```html
{{ synthesis.input_text }}
```

Add CSS for text overflow handling if needed.

---

### Phase 6: JIT Warmup on Container Startup

**Approach:** Add warmup to FastAPI lifespan, no DB record created (direct TTS service call). Delete warmup audio file after completion.

**File:** `src/presentation/api/main.py`

In `lifespan()` function, after DB init:
```python
# JIT Warmup
if os.environ.get("WARMUP_ON_STARTUP", "true").lower() == "true":
    logger.info("Starting JIT warmup synthesis...")
    try:
        tts_service = get_tts_service()

        # Run warmup synthesis with Arabic text
        result = await tts_service.synthesize(
            text="مرحبا، هذا اختبار تسخين النظام",
            model="multilingual",
            language="ar",
            cfg_weight=0.5,
            exaggeration=0.5
        )

        logger.info(f"JIT warmup completed in {result['processing_time_seconds']:.2f}s")

        # Note: This is a direct TTS call, no DB record created
        # The synthesize method returns audio_data bytes directly (not saved to disk)
        # No cleanup needed - audio_data is in memory only

    except Exception as e:
        logger.warning(f"JIT warmup failed: {e}")
```

**Important:** The `tts_service.synthesize()` method returns `audio_data` as bytes in memory. It does NOT save to disk or create a DB record. The use case layer (`SynthesizeTextUseCase`) is what saves files and creates records. By calling the service directly, we avoid any file/DB artifacts.

**File:** `docker-compose.yml`

Update health check start_period to allow warmup time:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8002/api/v1/health"]
  interval: 30s
  timeout: 10s
  retries: 5
  start_period: 120s  # Increased from 60s for warmup
```

**File:** `src/presentation/api/.env.example`

Add:
```bash
# JIT Warmup on Container Startup
WARMUP_ON_STARTUP=true
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/presentation/api/main.py` | UTF-8 JSON response, JIT warmup in lifespan |
| `src/infrastructure/config/settings.py` | Default model to multilingual |
| `src/presentation/api/schemas/synthesis_schema.py` | VoiceReferenceInfo schema, default model |
| `src/application/dto/synthesis_dto.py` | VoiceReferenceInfoDTO |
| `src/infrastructure/persistence/repositories/sqlite_synthesis_repository.py` | Eager load voice_reference |
| `src/application/use_cases/get_all_syntheses_use_case.py` | Pass voice ref info to DTO |
| `src/presentation/api/.env.example` | Default model, WARMUP_ON_STARTUP |
| `docker-compose.yml` | Default model, health check start_period |
| `synthesis-history.component.ts` | Pagination logic, remove truncation |
| `synthesis-history.component.html` | Pagination UI, voice ref display, full text |
| `synthesis-history.component.css` | Pagination styles |
| `synthesis.model.ts` | VoiceReferenceInfo interface |
| `synthesize.component.ts` | Default model selection |

---

## Implementation Order

1. **Phase 1:** Arabic encoding fix (blocks all testing)
2. **Phase 2:** Default model change (simple, low risk)
3. **Phase 3:** Voice reference info (backend then frontend)
4. **Phase 4:** Pagination (frontend only)
5. **Phase 5:** Full text display (frontend only)
6. **Phase 6:** JIT warmup (requires container restart to test)

---

## Testing Plan

### Phase 1 Tests: Arabic Text Encoding

**Test 1.1: Backend API Response (curl)**
```bash
# Create synthesis with Arabic text
curl -s -X POST http://localhost:8002/api/v1/syntheses \
  -H "Content-Type: application/json" \
  -d '{"text": "مرحبا بكم في توكلنا", "model": "multilingual", "language": "ar"}' \
  | python -m json.tool

# Expected: input_text shows Arabic characters, NOT "????????"
```

**Test 1.2: Get Syntheses List**
```bash
curl -s http://localhost:8002/api/v1/syntheses | python -m json.tool
# Expected: All Arabic text displays correctly in response
```

**Test 1.3: Frontend Display**
- Open http://localhost:4201
- Create synthesis with Arabic text
- Navigate to History view
- **Expected:** Arabic text displays correctly (not garbled)

---

### Phase 2 Tests: Default Model

**Test 2.1: Backend Default**
```bash
# Create synthesis without specifying model
curl -s -X POST http://localhost:8002/api/v1/syntheses \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "en"}' \
  | python -m json.tool

# Expected: "model": "multilingual"
```

**Test 2.2: Frontend Default**
- Open http://localhost:4201
- Navigate to Synthesize page
- **Expected:** Model dropdown shows "multilingual" selected by default

**Test 2.3: Docker Preload Order**
```bash
docker logs chatterbox-backend 2>&1 | grep -i "preload\|model"
# Expected: multilingual model loaded first
```

---

### Phase 3 Tests: Voice Reference Info

**Test 3.1: Syntheses List with Voice Reference**
```bash
# Get syntheses list
curl -s http://localhost:8002/api/v1/syntheses | python -m json.tool

# Expected: Each synthesis with voice_reference_id should have:
# "voice_reference": {
#   "id": "...",
#   "name": "Ahmad",
#   "language": "ar"
# }
```

**Test 3.2: Frontend History Display**
- Open http://localhost:4201
- Navigate to History view
- Find a synthesis that used a voice reference
- **Expected:** Shows "Voice: Ahmad (AR)" or similar

**Test 3.3: Synthesis Without Voice Reference**
```bash
# Create synthesis without voice reference
curl -s -X POST http://localhost:8002/api/v1/syntheses \
  -H "Content-Type: application/json" \
  -d '{"text": "Test", "model": "multilingual", "language": "en"}' \
  | python -m json.tool

# Expected: "voice_reference": null
```

---

### Phase 4 Tests: Pagination

**Test 4.1: API Pagination**
```bash
# Get first page (10 items)
curl -s "http://localhost:8002/api/v1/syntheses?limit=10&offset=0" | python -m json.tool
# Expected: syntheses array with max 10 items, total shows full count

# Get second page
curl -s "http://localhost:8002/api/v1/syntheses?limit=10&offset=10" | python -m json.tool
# Expected: Next 10 items (or remaining items)
```

**Test 4.2: Frontend Pagination Controls**
- Open http://localhost:4201
- Navigate to History view
- **Expected:**
  - Shows 10 items per page
  - "Page 1 of X (Y total)" displayed
  - Previous button disabled on first page
  - Next button works to go to page 2
  - Previous button enabled on page 2

**Test 4.3: Edge Cases**
- Create 15+ syntheses
- Test navigation through all pages
- **Expected:** All syntheses accessible via pagination

---

### Phase 5 Tests: Full Text Display

**Test 5.1: Long Text in History**
- Create synthesis with text > 100 characters
- Navigate to History view
- **Expected:** Full text displayed, no "..." truncation

**Test 5.2: Arabic Long Text**
- Create synthesis with long Arabic text (Tawakkalna description)
- Navigate to History view
- **Expected:** Full Arabic text displayed with RTL alignment

---

### Phase 6 Tests: JIT Warmup

**Test 6.1: Container Startup Logs**
```bash
# Restart container and watch logs
docker compose restart backend
docker logs -f chatterbox-backend 2>&1 | grep -i "warmup"

# Expected:
# "Starting JIT warmup synthesis..."
# "JIT warmup completed in X.XXs"
```

**Test 6.2: First Request Performance**
```bash
# After container restart, time first synthesis
time curl -s -X POST http://localhost:8002/api/v1/syntheses \
  -H "Content-Type: application/json" \
  -d '{"text": "Test", "model": "multilingual", "language": "en"}'

# Expected: Response time < 10s (not 30s+ like before warmup)
```

**Test 6.3: No Warmup Artifacts**
```bash
# Check database for warmup records
docker exec chatterbox-backend python3 -c "
from src.infrastructure.persistence.database import SessionLocal
from src.infrastructure.persistence.models.synthesis_model import SynthesisModel
db = SessionLocal()
count = db.query(SynthesisModel).filter(SynthesisModel.input_text.contains('تسخين')).count()
print(f'Warmup records in DB: {count}')
"

# Expected: Warmup records in DB: 0
```

**Test 6.4: Warmup Disabled**
```bash
# Set WARMUP_ON_STARTUP=false and restart
# Expected: No warmup logs, first request has JIT delay
```

---

### Integration Tests

**Test I.1: Full Workflow**
1. Restart container (triggers warmup)
2. Create synthesis with Arabic text + voice reference
3. Navigate to History
4. Verify: Arabic displays, voice ref shows, pagination works, full text shown
5. First synthesis should be fast (warmed up)

**Test I.2: Multiple Pages with Voice References**
1. Create 15+ syntheses (mix with/without voice references)
2. Navigate through pagination
3. Verify voice reference info shows on all pages

---

## Testing Checklist

- [ ] Arabic text displays correctly in curl and frontend
- [ ] Multilingual is default model in UI and backend
- [ ] Voice reference name/language shown in history
- [ ] Pagination works with 10 items per page
- [ ] Full text displayed without truncation
- [ ] Container startup includes JIT warmup
- [ ] First synthesis request is fast (no JIT delay)
- [ ] No warmup artifacts in DB or filesystem

---

## TODOs

- [ ] Phase 1: Fix Arabic text encoding
- [ ] Phase 2: Default model change to multilingual
- [ ] Phase 3: Voice reference info in synthesis response
- [ ] Phase 4: Pagination (10 items per page)
- [ ] Phase 5: Show full text (remove truncation)
- [ ] Phase 6: JIT warmup on container startup
- [ ] Testing and verification
