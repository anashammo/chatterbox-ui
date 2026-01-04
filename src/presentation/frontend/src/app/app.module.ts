import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';

// Feature Components
import { SynthesizeComponent } from './features/synthesize/synthesize.component';
import { SynthesisHistoryComponent } from './features/synthesis-history/synthesis-history.component';
import { SynthesisDetailComponent } from './features/synthesis-detail/synthesis-detail.component';
import { VoiceReferencesComponent } from './features/voice-references/voice-references.component';

// Shared Components
import { PopupComponent } from './shared/components/popup/popup.component';
import { FooterComponent } from './shared/components/footer/footer.component';

// Core Services
import { ApiService } from './core/services/api.service';
import { SynthesisService } from './core/services/synthesis.service';

// Shared Services
import { PopupService } from './shared/services/popup.service';

@NgModule({
  declarations: [
    AppComponent,
    SynthesizeComponent,
    SynthesisHistoryComponent,
    SynthesisDetailComponent,
    VoiceReferencesComponent,
    PopupComponent,
    FooterComponent
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    CommonModule,
    AppRoutingModule,
    HttpClientModule,
    FormsModule
  ],
  providers: [
    ApiService,
    SynthesisService,
    PopupService
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
