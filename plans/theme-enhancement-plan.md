# Theme Enhancement Plan - Dark Mode Redesign

## Objective
Update the Chatterbox TTS frontend to match the dark theme design from the Whisper transcription app reference screenshot.

## Current State Analysis

### Problem
The current UI has an inconsistent theme:
- Body background is dark (`#1a202c`)
- But cards/forms use white (`#fff`) - looks jarring
- Text colors assume light backgrounds (`#333`)
- Status badges use light pastels

### Reference Theme (Target)
From the Whisper app screenshot:
- **Background**: Deep navy `#1a1d29`
- **Cards**: Dark slate `#242836` with subtle border `#3a4055`
- **Headings**: Near white `#f7fafc`
- **Body text**: Light gray `#e2e8f0`
- **Labels/muted**: Gray `#718096`
- **Primary button**: Blue `#4f6594` / `#5a7ab8`
- **Secondary button**: Orange/coral `#e07b5a`
- **Status badges**: Dark backgrounds with colored text
  - Completed: Green `#10b981` on `#064e3b`
  - Processing: Blue `#3b82f6` on `#1e3a5f`
  - Failed: Red `#ef4444` on `#7f1d1d`
  - Enhanced/Special: Purple/magenta `#a855f7` on `#4c1d95`
- **Input fields**: Dark `#1e2230` with border `#3a4055`
- **Footer**: Dark `#1a1d29` with colored links

## Files to Modify

### 1. Global Styles
- `src/presentation/frontend/src/styles.css`
  - Update body background
  - Define CSS custom properties (variables) for theme colors
  - Update scrollbar colors
  - Add input/textarea dark styles

### 2. Component Styles (ALL need dark theme updates)
- `src/app/app.component.css` - Container background
- `src/app/features/synthesize/synthesize.component.css` - Form card, inputs, buttons
- `src/app/features/synthesis-history/synthesis-history.component.css` - List cards, badges
- `src/app/features/synthesis-detail/synthesis-detail.component.css` - Detail view, meta sections
- `src/app/features/voice-references/voice-references.component.css` - Upload form, voice cards
- `src/app/shared/components/footer/footer.component.css` - Footer styling (already dark)

## Implementation Strategy

### Phase 1: CSS Custom Properties (Variables)
Add to `styles.css`:
```css
:root {
  /* Backgrounds */
  --bg-primary: #1a1d29;
  --bg-card: #242836;
  --bg-input: #1e2230;
  --bg-hover: #2d3348;

  /* Borders */
  --border-color: #3a4055;
  --border-subtle: #2d3348;

  /* Text */
  --text-primary: #f7fafc;
  --text-secondary: #e2e8f0;
  --text-muted: #718096;
  --text-label: #a0aec0;

  /* Accent Colors */
  --accent-blue: #5a7ab8;
  --accent-blue-hover: #4f6594;
  --accent-orange: #e07b5a;
  --accent-orange-hover: #c9684a;
  --accent-green: #10b981;
  --accent-purple: #a855f7;
  --accent-red: #ef4444;

  /* Status Badges */
  --badge-completed-bg: #064e3b;
  --badge-completed-text: #10b981;
  --badge-processing-bg: #1e3a5f;
  --badge-processing-text: #3b82f6;
  --badge-failed-bg: #7f1d1d;
  --badge-failed-text: #ef4444;
  --badge-pending-bg: #78350f;
  --badge-pending-text: #fbbf24;
  --badge-special-bg: #4c1d95;
  --badge-special-text: #a855f7;
}
```

### Phase 2: Global Element Styling
Update `styles.css`:
- Body background to `var(--bg-primary)`
- Default text color to `var(--text-secondary)`
- Headings to `var(--text-primary)`
- Links default color
- Form elements (inputs, textareas, selects) dark styling

### Phase 3: Component Updates
For each component CSS file:
1. Replace all `white`/`#fff` backgrounds with `var(--bg-card)`
2. Replace light grays (`#f8f9fa`) with `var(--bg-input)` or `var(--bg-hover)`
3. Replace dark text (`#333`) with `var(--text-primary)` or `var(--text-secondary)`
4. Replace muted text (`#666`) with `var(--text-muted)`
5. Update button colors to match reference
6. Update status badges to dark style
7. Add subtle borders where needed

### Phase 4: Special Elements
- Model badges: Use purple accent like "ENHANCED" badge
- Action buttons: Orange for secondary actions
- Cards: Add `border: 1px solid var(--border-color)`
- Inputs: Focus state with blue glow

## Color Palette Summary

| Element | Current | New |
|---------|---------|-----|
| Page background | #1a202c | #1a1d29 |
| Card background | #ffffff | #242836 |
| Input background | #ffffff | #1e2230 |
| Primary text | #333333 | #f7fafc |
| Secondary text | #666666 | #e2e8f0 |
| Muted text | #666666 | #718096 |
| Primary button | #3498db | #5a7ab8 |
| Secondary button | #ecf0f1 | #e07b5a |
| Border color | #e0e0e0 | #3a4055 |
| Completed badge | #d4edda/#155724 | #064e3b/#10b981 |

## Testing Strategy

1. **Visual Testing**:
   - Open frontend at http://localhost:4201
   - Check all pages: Synthesize, History, Detail, Voice References
   - Verify contrast and readability
   - Test status badge visibility for all states
   - Verify button hover states

2. **Cross-Browser**:
   - Chrome (primary)
   - Firefox
   - Edge

3. **Responsive**:
   - Desktop (1920x1080)
   - Tablet (768px)
   - Mobile (375px)

## Risks & Mitigations

1. **Contrast Issues**: Use WebAIM contrast checker - all text must meet WCAG AA (4.5:1)
2. **CSS Variable Support**: IE11 doesn't support - acceptable as modern browsers only
3. **Regression**: Test all interactive elements (buttons, inputs, dropdowns)

## Rollback Plan
- All changes are CSS only - easy to revert via git
- Keep original color values commented for reference during transition

## TODOs

- [x] Create CSS custom properties in styles.css
- [x] Update global styles (body, scrollbar, inputs)
- [x] Update app.component.css
- [x] Update synthesize.component.css
- [x] Update synthesis-history.component.css
- [x] Update synthesis-detail.component.css
- [x] Update voice-references.component.css
- [x] Update footer.component.css
- [x] Update popup.component.css
- [ ] Test all pages visually
- [ ] Verify all status badge states
- [ ] Test responsive layout

## Implementation Complete
All CSS files have been updated with the new dark theme using CSS custom properties.
