import { Component, ChangeDetectorRef  } from '@angular/core';

@Component({
  selector: 'app-game-card',
  standalone: true,
  imports: [],
  templateUrl: './game-card.component.html',
  styleUrl: './game-card.component.css'
})
export class GameCardComponent {
  price = "$0.00";
  game_name = "Test";
  thumbnail_image: string = '../../../assets/No-Image-Available.png';

  handleCardClick() {
    console.log('Card clicked!');
  }
}
