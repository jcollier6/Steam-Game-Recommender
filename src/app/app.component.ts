import { Component, OnInit } from '@angular/core';
import { GameService, Recommended_Game, Recent_Game } from './services/game.service';
import { HttpClientModule } from '@angular/common/http';
import { HeroBannerComponent } from './hero-banner/hero-banner.component';


@Component({
  selector: 'app-root',
  standalone: true,
  templateUrl: './app.component.html',
  imports: [ HttpClientModule, HeroBannerComponent],
  providers: [ HttpClientModule, GameService ],
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  recommendedGames: Recommended_Game[] = [];
  recentGames: Recent_Game[] = [];
  recommendedGameListExist = false;
  recentGameListExist = false;

  constructor(private gameService: GameService) {}

  ngOnInit(): void {
    this.gameService.getRecommendedGames().subscribe((data) => {
      this.recommendedGames = data;
      this.recommendedGameListExist = this.recommendedGames.length > 0;
    });

    this.gameService.getRecentlyPlayed().subscribe((data) => {
      this.recentGames = data;
      this.recentGameListExist = this.recentGames.length > 0;
    });
  }
}

