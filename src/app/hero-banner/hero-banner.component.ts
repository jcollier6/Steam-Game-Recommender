import { Component } from '@angular/core';
import { Router } from '@angular/router';


@Component({
  selector: 'app-hero-banner',
  standalone: true,
  imports: [],
  templateUrl: './hero-banner.component.html',
  styleUrl: './hero-banner.component.css'
})
export class HeroBannerComponent {

  constructor(private router: Router) {}

  goToHome() {
    this.router.navigate(['/home-page']);
  }

  signOut() {
    this.router.navigate(['/enter-steam-id']);
  }
}
