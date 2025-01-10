import { Component, OnInit } from '@angular/core';
import { GameService, Game } from './../services/game.service';
import { HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-hero-banner',
  standalone: true,
  imports: [],
  templateUrl: './hero-banner.component.html',
  styleUrl: './hero-banner.component.css'
})
export class HeroBannerComponent {
  games: Game[] = [];
  gameListExist = false;

  constructor(private gameService: GameService) {}

  ngOnInit(): void {
    this.gameService.getRecommendedGames().subscribe((data) => {
      this.games = data;
      if( data.length > 0){
        this.gameListExist = true;
      }
      console.log("in sub", this.games)
    });
  }
}
