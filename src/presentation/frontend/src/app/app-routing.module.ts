import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { SynthesizeComponent } from './features/synthesize/synthesize.component';
import { SynthesisHistoryComponent } from './features/synthesis-history/synthesis-history.component';
import { SynthesisDetailComponent } from './features/synthesis-detail/synthesis-detail.component';
import { VoiceReferencesComponent } from './features/voice-references/voice-references.component';

const routes: Routes = [
  { path: '', component: SynthesizeComponent },
  { path: 'synthesis/:id', component: SynthesisDetailComponent },
  { path: 'history', component: SynthesisHistoryComponent },
  { path: 'voices', component: VoiceReferencesComponent },
  { path: '**', redirectTo: '' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes, {
    onSameUrlNavigation: 'ignore',
    urlUpdateStrategy: 'deferred'
  })],
  exports: [RouterModule]
})
export class AppRoutingModule { }
