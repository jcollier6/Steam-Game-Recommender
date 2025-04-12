import { Component } from '@angular/core';
import { GameService } from './services/game-service/game.service';
import { HttpClientModule } from '@angular/common/http';
import { RouterModule } from '@angular/router';
import { BottomInfoBarComponent } from "./components/bottom-info-bar/bottom-info-bar.component";
import { NavigationBarComponent } from './components/navigation-bar/navigation-bar.component';

@Component({
  selector: 'app-root',
  standalone: true,
  templateUrl: './app.component.html',
  imports: [HttpClientModule, RouterModule, BottomInfoBarComponent, NavigationBarComponent],
  providers: [ HttpClientModule, GameService ],
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  
}

