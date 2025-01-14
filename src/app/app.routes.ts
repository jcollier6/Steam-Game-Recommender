import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { EnterSteamIdComponent } from './enter-steam-id/enter-steam-id.component';
import { HomePageComponent } from './home-page/home-page.component';


export const routes: Routes = [
    { path: '', redirectTo: '/enter-steam-id', pathMatch: 'full' },
    { path: 'enter-steam-id', component: EnterSteamIdComponent },
    { path: 'home-page', component: HomePageComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
