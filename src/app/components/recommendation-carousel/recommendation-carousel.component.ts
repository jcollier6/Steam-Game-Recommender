import { Component, Input, Output, EventEmitter, ElementRef, ViewChild, Renderer2 } from '@angular/core';
import { Game_Info } from '../../services/game-service/game.service';
import { CommonModule } from '@angular/common';
import { trigger, state, style, animate, transition } from '@angular/animations';


@Component({
  selector: 'app-recommendation-carousel',
  standalone: true,
  imports: [ CommonModule ],
  providers: [ ],
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
  recommendedGameListExist = false;
  currentIndex = 0;
  currentGame: Game_Info = {
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

  tags: string[] = [];
  tagsExist: boolean = false;

  private _recommendedGames: Game_Info[] = [];

  @Output() viewAllClicked = new EventEmitter<string>();

  @Input()
  set recommendedGames(value: Game_Info[]) {
    this._recommendedGames = value || [];
    this.recommendedGameListExist = this._recommendedGames.length > 0;
    this.updateCurrentGame();
  }
  
  get recommendedGames(): Game_Info[] {
    return this._recommendedGames;
  }

  constructor(private renderer: Renderer2) {}

  handleCardClick(app_id: string) {
    const url = `https://store.steampowered.com/app/${app_id}`;
    window.open(url, '_blank');
  }

  handleViewAllClick() {
    this.viewAllClicked.emit('Recommendations')
  }
  
  updateCurrentGame(): void {
    this.currentGame = this.recommendedGames[this.currentIndex];
    this.tagsExist = this.currentGame['tags'] && this.currentGame['tags'].length > 0;
    if (this.tagsExist) {
      this.renderTags(this.currentGame.tags);
    }

    this.header_image = this.currentGame['header_image'] || '../../assets/No-Image-Available.png';
    this.price = this.currentGame['is_free'] ? 'Free To Play' : String(this.currentGame['price_usd'] || "");
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
  
  @ViewChild('tagsHolder', { static: true }) tagsHolder!: ElementRef<HTMLDivElement>;
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
