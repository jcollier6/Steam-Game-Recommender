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
  constructor(private httpClient: HttpClient) {}

  getAllTags(): Observable<string[]> {
    return this.httpClient.get<string[]>(`/all_tags`);
  }

  getRecommendedGames(steam_id: string, top_k: number = 10): Observable<Game_Info[]> {
    return this.httpClient.get<Game_Info[]>(
      `/recommended_games?steam_id=${steam_id}&top_k=${top_k}`
    );
  }

  getAllRecommendedGames(): Observable<Game_Info[]> {
    return this.httpClient.get<Game_Info[]>(`/all_recommended_games`);
  }

  getRecentlyPlayed(): Observable<Game_Info[]> {
    return this.httpClient.get<Game_Info[]>(`/recently_played`);
  }

  getTopTagGames(): Observable<Record<string, Game_Info[]>> {
    return this.httpClient.get<Record<string, Game_Info[]>>(`/top_tag_games`);
  }

  getSteamTagCounts(): Observable<Record<string,number>> {
    return this.httpClient.get<Record<string,number>>(`/steam_tag_counts`);
  }

  getSteamReleaseYears(): Observable<Record<string,number>> {
    return this.httpClient.get<Record<string,number>>(`/steam_release_years`);
  }
}
