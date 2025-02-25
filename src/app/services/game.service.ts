import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Recommended_Game {
  app_id: string;
  name: string;
  is_free: boolean;
  price_usd: number;
  tags: string[];
  header_image: string;
  screenshots: string[];
}

export interface Recent_Game {
  app_id: string;
  name: string;
  playtime_forever: number;
  playtime_2weeks: number;
}

@Injectable({
  providedIn: 'root'
})
export class GameService {
  private apiUrl = 'http://localhost:8000';

  constructor(private httpClient: HttpClient) {}

  getRecommendedGames(): Observable<Recommended_Game[]> {
    return this.httpClient.get<Recommended_Game[]>(`${this.apiUrl}/recommended_games`);
  }

  getRecentlyPlayed(): Observable<Recent_Game[]> {
    return this.httpClient.get<Recent_Game[]>(`${this.apiUrl}/recently_played`);
  }
}
