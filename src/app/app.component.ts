import { Component } from '@angular/core';
import { GameService } from './services/game.service';
import { HttpClientModule } from '@angular/common/http';
import { HeroBannerComponent } from './hero-banner/hero-banner.component';
import { RouterModule } from '@angular/router';


@Component({
  selector: 'app-root',
  standalone: true,
  templateUrl: './app.component.html',
  imports: [ HttpClientModule, HeroBannerComponent, RouterModule],
  providers: [ HttpClientModule, GameService ],
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  
}

