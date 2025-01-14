import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { HttpClient, HttpHeaders } from '@angular/common/http';
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

  constructor(private router: Router, private http: HttpClient) {}

  onSubmit() {
    if (this.steamId.trim()) {
      console.log('Steam ID submitted:', this.steamId);

      const url = 'http://localhost:8000/submit-steam-id'; 
      const body = { steamId: this.steamId };
      const headers = new HttpHeaders({ 'Content-Type': 'application/json' });

      this.http.post(url, body, { headers }).subscribe({
        next: (response) => {
          console.log('Response from server:', response);
          this.router.navigate(['/home-page']);
        },
        error: (error) => {
          console.error('Error submitting Steam ID:', error);
          alert(`Error: ${error.error.detail}`);
        }
      });
    } else {
      console.error('Steam ID is required.');
    }
  }
}
