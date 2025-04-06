import { Component, Input, ElementRef, ViewChild, Renderer2 } from '@angular/core';
import { Game_Info } from '../../services/game.service';
import { CommonModule } from '@angular/common';


@Component({
  selector: 'app-game-card',
  standalone: true,
  imports: [ CommonModule ],
  templateUrl: './game-card.component.html',
  styleUrl: './game-card.component.css'
})
export class GameCardComponent {
  price: string = '';
  game_name = '';
  thumbnail_image: string = '../../../assets/No-Image-Available.png';

  handleCardClick(app_id: string) {
    const url = `https://store.steampowered.com/app/${app_id}`;
    window.open(url, '_blank');
  }

  constructor(
    private renderer: Renderer2,
  ) {}

  header_image: string = '../../assets/No-Image-Available.png';
  tagsExist: boolean = false;
  gameCard: Game_Info = {
      app_id: '',
      name: '',
      is_free: false,
      price_usd: 0,
      tags: [],
      header_image: '../../assets/No-Image-Available.png',
      screenshots: []
    };

  @Input()
  set game(value: {app_id: string, name: string, is_free: boolean, price_usd: number, tags: string[], header_image: string, screenshots: string[]}) {
    this.gameCard = value;
    this.renderGame();
  }

  renderGame() {
    this.tagsExist = this.gameCard['tags'] && this.gameCard['tags'].length > 0;
    if (this.tagsExist) {
      this.renderTags(this.gameCard.tags);
    }

    this.header_image = this.gameCard['header_image'];
    this.price = this.gameCard['is_free'] ? 'Free To Play' : String(this.gameCard['price_usd'] || "");
  }

  @ViewChild('tagsHolder', { static: true }) tagsHolder!: ElementRef<HTMLDivElement>;

  renderTags(tags: string[]): void {
    const container = this.tagsHolder.nativeElement;
    container.innerHTML = '';

    for (const tagText of tags) {
      const tag = this.renderer.createElement('span');
      this.renderer.addClass(tag, 'tag');
      const text = this.renderer.createText(tagText);
      this.renderer.appendChild(tag, text);
      this.renderer.appendChild(container, tag);
    }
  }
}
