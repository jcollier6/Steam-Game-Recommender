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

  getAllTags(): Observable<string[]> {
    return this.httpClient.get<string[]>(`${this.apiUrl}/all_tags`);
  }

  getRecommendedGames(): Observable<Game_Info[]> {
    return this.httpClient.get<Game_Info[]>(`${this.apiUrl}/recommended_games`);
  }

  getAllRecommendedGames(): Observable<Game_Info[]> {
    return this.httpClient.get<Game_Info[]>(`${this.apiUrl}/all_recommended_games`);
  }

  getRecentlyPlayed(): Observable<Game_Info[]> {
    return this.httpClient.get<Game_Info[]>(`${this.apiUrl}/recently_played`);
  }

  getTopTagGames(): Observable<Record<string, Game_Info[]>> {
    return this.httpClient.get<Record<string, Game_Info[]>>(`${this.apiUrl}/top_tag_games`);
  }

  getSteamTagCounts(): Observable<Record<string,number>> {
    return this.httpClient.get<Record<string,number>>(`${this.apiUrl}/steam_tag_counts`);
  }
}
