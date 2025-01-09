import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Game {
  app_id: string;
  name: string;
  overlap_score: number;
}

@Injectable({
  providedIn: 'root'
})
export class GameService {
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  getRecommendedGames(): Observable<Game[]> {
    return this.http.get<Game[]>(`${this.apiUrl}/recommended_games`);
  }
}
