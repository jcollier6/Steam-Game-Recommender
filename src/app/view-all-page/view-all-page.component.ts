import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { GameCardComponent } from "../components/game-card/game-card.component";
import { GameService, Game_Info } from '../services/game-service/game.service';
import { ShellLoaderComponent } from "../components/shell-loader/shell-loader.component";
import { ActivatedRoute } from '@angular/router';
import { CustomSliderComponent } from "../components/custom-slider/custom-slider.component";
import { FormControl, FormGroup, ReactiveFormsModule } from '@angular/forms';


@Component({
  selector: 'view-all',
  standalone: true,
  imports: [ GameCardComponent, ShellLoaderComponent, CustomSliderComponent, ReactiveFormsModule ],
  templateUrl: './view-all-page.component.html',
  styleUrl: './view-all-page.component.css'
})
export class ViewAllPageComponent implements OnInit {
  allRecommendedGames: Game_Info[] = [];
  cardGroup: string | null = '';
  isReady = false;
  priceSliderValue:number = 0;
  reviewCountSliderValue: number = 0;
  reviewScoreSliderValue: number = 0;
  isRecommendationPage: boolean = false;
  isRecentlyPlayedPage: boolean = false;

  checkBox = new FormGroup({
    isHideF2PChecked: new FormControl(false),
    isIndieOnlyChecked: new FormControl(false)
  });

  constructor(
    private gameService: GameService,
    private route: ActivatedRoute,
    private cdRef: ChangeDetectorRef
  ) {}


  ngAfterViewInit(): void {
    this.isReady = true;
    this.cdRef.detectChanges();
  }

  ngOnInit(): void {
    this.route.queryParamMap.subscribe(params => {
      this.cardGroup = params.get('cardGroup');
      if(this.cardGroup == 'Recommendations') {
        this.isRecommendationPage = true;
      } else if (this.cardGroup == 'Recently Played') {
        this.isRecentlyPlayedPage = true;
      }
    });
    

    this.gameService.getRecommendedGames().subscribe((data) => {
      this.allRecommendedGames = data;
    });
  }




  
}
