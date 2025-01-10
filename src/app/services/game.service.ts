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

  constructor(private httpClient: HttpClient) {}

  getRecommendedGames(): Observable<Game[]> {
    console.log("getRecommenedGames");
    return this.httpClient.get<Game[]>(`${this.apiUrl}/recommended_games`);
  }
}
