import { Component, ElementRef, ViewChild, Renderer2 } from '@angular/core';
import { GameService, Recommended_Game } from '../services/game.service';
import { CommonModule } from '@angular/common';


@Component({
  selector: 'app-recommendation-carousel',
  standalone: true,
  imports: [ CommonModule ],
  providers: [ GameService ],
  templateUrl: './recommendation-carousel.component.html',
  styleUrl: './recommendation-carousel.component.css'
})
export class RecommendationCarouselComponent {
  recommendedGames: Recommended_Game[] = [];
  recommendedGameListExist = false;
  currentIndex = 0;
  currentGame: Recommended_Game = {
    app_id: '',
    name: '',
    price: 0,
    tags: [],
    thumbnail: '',
    screenshots: []
  };


  tags: string[] = ['Open World Survival Craft', 'PvE', 'Survival', 'Multiplayer', 'Co-op'];
  tagsExist: boolean = false;

  @ViewChild('tagsHolder', { static: true }) tagsHolder!: ElementRef<HTMLDivElement>;

  constructor(
    private renderer: Renderer2,
    private gameService: GameService) {}

  ngOnInit(): void {
    this.gameService.getRecommendedGames().subscribe((data) => {
      this.recommendedGames = data;
      console.log(data)
      if (this.recommendedGames.length > 0) {
        this.recommendedGameListExist = true;
        this.currentIndex = 0;
        this.updateCurrentGame();
      }
      else {
        this.recommendedGameListExist = false;
      }
    });
  }

  updateCurrentGame(): void {
    this.currentGame = this.recommendedGames[this.currentIndex];
    this.tagsExist = this.currentGame['tags'] && this.currentGame['tags'].length > 0;
    if (this.tagsExist) {
      this.renderTags(this.currentGame.tags);
    }
  }
  
  changeGame(direction: 'next' | 'previous'): void {
    if (direction === 'next') {
      this.currentIndex = (this.currentIndex + 1) % this.recommendedGames.length;
    } else if (direction === 'previous') {
      this.currentIndex = (this.currentIndex - 1 + this.recommendedGames.length) % this.recommendedGames.length;
    }
    this.updateCurrentGame();
  }
  

  renderTags(tags: string[]): void {
    const container = this.tagsHolder.nativeElement;
    container.innerHTML = ''; 

    tags.forEach(tagText => {
      const tag = this.renderer.createElement('span');
      this.renderer.addClass(tag, 'tag');
      const text = this.renderer.createText(tagText);
      this.renderer.appendChild(tag, text);
      this.renderer.appendChild(container, tag);     
    });
  }
}
