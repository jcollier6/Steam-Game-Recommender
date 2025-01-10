import { Component, OnInit } from '@angular/core';
import { GameService, Game } from './services/game.service';
import { HttpClientModule } from '@angular/common/http';


@Component({
  selector: 'app-root',
  standalone: true,
  templateUrl: './app.component.html',
  imports: [ HttpClientModule ],
  providers: [ HttpClientModule, GameService ],
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
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

