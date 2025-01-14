import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { FormsModule } from '@angular/forms';

@Component({
  standalone: true,
  selector: 'app-enter-steam-id',
  templateUrl: './enter-steam-id.component.html',
  styleUrls: ['./enter-steam-id.component.css'],
  imports: [FormsModule]
})
export class EnterSteamIdComponent {
  steamId: string = '';

  constructor(private router: Router) {}

  onSubmit() {
    if (this.steamId.trim()) {
      console.log('Steam ID submitted:', this.steamId);
      this.router.navigate(['/home-page']);
    } else {
      console.error('Steam ID is required.');
    }
  }
}
