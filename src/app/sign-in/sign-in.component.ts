import { ChangeDetectionStrategy, Component, ChangeDetectorRef } from '@angular/core';
import { ReactiveFormsModule, FormControl, Validators, FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Router } from '@angular/router';


@Component({
  selector: 'app-sign-in',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, FormsModule],
  templateUrl: './sign-in.component.html',
  styleUrl: './sign-in.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SignInComponent {
  idInput = new FormControl('', [
    Validators.required, 
    Validators.pattern('^[0-9]+$')
  ]);

  get isInvalid() {
    return this.idInput.invalid && this.idInput.touched;
  }

  steamId: string = '';
    loading = false;
  
    constructor(
      private http: HttpClient, 
      private router: Router,
      private cdr: ChangeDetectorRef) {}
    
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
            this.cdr.detectChanges();
          },
          complete: () => {
            this.loading = false;
            this.cdr.detectChanges();
          }
        });
      } else {
        console.error('Steam ID is required.');
      }
    }
}
