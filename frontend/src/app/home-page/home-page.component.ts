import { Component, ChangeDetectorRef, OnInit } from '@angular/core';
import { GameService, Game_Info } from '../services/game-service/game.service';
import { HttpClientModule } from '@angular/common/http';
import { RouterModule, Router } from '@angular/router';
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
  recommendedGames: Game_Info[] = [];
  recentGames: Game_Info[] = [];
  tagNames: string[] = [];
  tagGames: Game_Info[][] = [];
  recommendedGameListExist = false;
  recentGameListExist = false;
  isReady = false;

  constructor(
    private gameService: GameService,
    private cdRef: ChangeDetectorRef,
    private router: Router
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
    });

    this.gameService.getTopTagGames().subscribe((data: Record<string, Game_Info[]>) => {
      this.tagNames = Object.keys(data);
      this.tagGames = this.tagNames.map((tag: string) => data[tag]);
    });
  }

  onViewAllClicked(cardGroup: string): void {
    localStorage.setItem('topTagNames', JSON.stringify([...this.tagNames]));
    this.router.navigate(['/view-all'], { queryParams: { cardGroup }});
  }
}