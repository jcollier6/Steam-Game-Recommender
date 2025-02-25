import { Component, ElementRef, ViewChild, Renderer2 } from '@angular/core';
import { GameService, Recommended_Game } from '../services/game.service';
import { CommonModule } from '@angular/common';
import { trigger, state, style, animate, transition } from '@angular/animations';


@Component({
  selector: 'app-recommendation-carousel',
  standalone: true,
  imports: [ CommonModule ],
  providers: [ GameService ],
  templateUrl: './recommendation-carousel.component.html',
  styleUrl: './recommendation-carousel.component.css',
  animations: [
    trigger('fadeAnimation', [
      state('visible', style({ 
        opacity: 1,
        filter: 'blur(0px)'
      })),
      state('hidden', style({ 
        opacity: 0,
        filter: 'blur(2px)' 
      })),
      transition('visible => hidden', animate('100ms ease-out')),
      transition('hidden => visible', animate('100ms ease-in'))
    ])
    
  ]
})
export class RecommendationCarouselComponent {
  fadeState: 'visible' | 'hidden' = 'visible';
  recommendedGames: Recommended_Game[] = [];
  recommendedGameListExist = false;
  currentIndex = 0;
  currentGame: Recommended_Game = {
    app_id: '',
    name: '',
    is_free: false,
    price_usd: 0,
    tags: [],
    header_image: '',
    screenshots: []
  };
  header_image: string = '../../assets/No-Image-Available.png';
  price: string = '';
  screenshots: string[] = ["", "", "", ""];

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

    this.header_image = this.currentGame['header_image'] || '../../assets/No-Image-Available.png';
    this.price = this.currentGame['is_free'] ? 'Free' : String(this.currentGame['price_usd'] || "");
    this.screenshots = this.currentGame.screenshots.map(url => url || "");
  }
  
  changeGame(direction: 'next' | 'previous'): void {
    this.fadeState = 'hidden';
    // Wait for the fade-out animation to complete
    setTimeout(() => {
      if (direction === 'next') {
        this.currentIndex = (this.currentIndex + 1) % this.recommendedGames.length;
      } else if (direction === 'previous') {
        this.currentIndex = (this.currentIndex - 1 + this.recommendedGames.length) % this.recommendedGames.length;
      }
      this.updateCurrentGame();
      this.fadeState = 'visible';
    }, 100);
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
