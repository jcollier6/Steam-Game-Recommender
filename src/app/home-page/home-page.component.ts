import { Component, ChangeDetectorRef, OnInit } from '@angular/core';
import { GameService, Recommended_Game, Recent_Game } from '../services/game.service';
import { HttpClientModule } from '@angular/common/http';
import { RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';
import { RecommendationCarouselComponent } from '../components/recommendation-carousel/recommendation-carousel.component';
import { GameCardHolderComponent } from "../components/game-card-holder/game-card-holder.component";
import { ShellLoaderComponent } from "../components/shell-loader/shell-loader.component";

@Component({
  selector: 'home-page',
  standalone: true,
  templateUrl: './home-page.component.html',
  imports: [HttpClientModule, RouterModule, RecommendationCarouselComponent, GameCardHolderComponent, ShellLoaderComponent, CommonModule],
  providers: [ HttpClientModule, GameService ],
  styleUrls: ['./home-page.component.css']
})
export class HomePageComponent implements OnInit {
  recommendedGames: Recommended_Game[] = [];
  recentGames: Recent_Game[] = [];
  recommendedGameListExist = false;
  recentGameListExist = false;
  isReady = false;

  constructor(
    private gameService: GameService,
    private cdRef: ChangeDetectorRef
  ) {}

  ngAfterViewInit(): void {
    this.isReady = true;
    this.cdRef.detectChanges();
  }
  

  ngOnInit(): void {
    this.gameService.getRecommendedGames().subscribe((data) => {
      this.recommendedGames = data;
    });

    this.gameService.getRecentlyPlayed().subscribe((data) => {
      this.recentGames = data;
      this.recentGameListExist = this.recentGames.length > 0;
    });
  }
}