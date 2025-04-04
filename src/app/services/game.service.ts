import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Game_Info {
  app_id: string;
  name: string;
  is_free: boolean;
  price_usd: number;
  tags: string[];
  header_image: string;
  screenshots: string[];
}

@Injectable({
  providedIn: 'root'
})
export class GameService {
  private apiUrl = 'http://localhost:8000';

  constructor(private httpClient: HttpClient) {}

  getRecommendedGames(): Observable<Game_Info[]> {
    return this.httpClient.get<Game_Info[]>(`${this.apiUrl}/recommended_games`);
  }

  getRecentlyPlayed(): Observable<Game_Info[]> {
    return this.httpClient.get<Game_Info[]>(`${this.apiUrl}/recently_played`);
  }
}
