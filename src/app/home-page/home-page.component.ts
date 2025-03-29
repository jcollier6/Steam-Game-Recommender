import { Component, OnInit } from '@angular/core';
import { GameService, Recommended_Game, Recent_Game } from '../services/game.service';
import { HttpClientModule } from '@angular/common/http';
import { RouterModule } from '@angular/router';
import { RecommendationCarouselComponent } from '../components/recommendation-carousel/recommendation-carousel.component';

@Component({
  selector: 'home-page',
  standalone: true,
  templateUrl: './home-page.component.html',
  imports: [HttpClientModule, RouterModule, RecommendationCarouselComponent],
  providers: [ HttpClientModule, GameService ],
  styleUrls: ['./home-page.component.css']
})
export class HomePageComponent implements OnInit {
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