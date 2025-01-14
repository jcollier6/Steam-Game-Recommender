import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';


@Component({
  standalone: true,
  selector: 'app-enter-steam-id',
  templateUrl: './enter-steam-id.component.html',
  styleUrls: ['./enter-steam-id.component.css'],
  imports: [CommonModule, FormsModule]
})
export class EnterSteamIdComponent {
  steamId: string = '';
  loading = false;

  constructor(private http: HttpClient, private router: Router) {}
  
  onSubmit() {
    if (this.steamId.trim()) {
      console.log('Steam ID submitted:', this.steamId);

      const url = 'http://localhost:8000/submit-steam-id'; 
      const body = { steamId: this.steamId };
      const headers = new HttpHeaders({ 'Content-Type': 'application/json' });

      this.loading = true;

      this.http.post(url, body, { headers }).subscribe({
        next: (response) => {
          console.log('Response from server:', response);
          this.router.navigate(['/home-page']);
        },
        error: (error) => {
          console.error('Error submitting Steam ID:', error);
          alert(`Error: ${error.error.detail}`);
          this.loading = false;
          this.steamId = '';
        },
        complete: () => {
          this.loading = false;
        }
      });
    } else {
      console.error('Steam ID is required.');
    }
  }
}
