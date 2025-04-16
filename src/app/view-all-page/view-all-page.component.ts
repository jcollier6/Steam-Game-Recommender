import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { GameCardComponent } from "../components/game-card/game-card.component";
import { GameService, Game_Info } from '../services/game-service/game.service';
import { ShellLoaderComponent } from "../components/shell-loader/shell-loader.component";
import { ActivatedRoute } from '@angular/router';
import { CustomSliderComponent } from "../components/custom-slider/custom-slider.component";
import { FormControl, FormGroup } from '@angular/forms';


@Component({
  selector: 'view-all',
  standalone: true,
  imports: [ GameCardComponent, ShellLoaderComponent, CustomSliderComponent ],
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
  topTagNames: string[] = [];
  selectedTags: Set<string> = new Set();
  filteredTags: string[] = [];

  checkBox = new FormGroup({
    isHideF2PChecked: new FormControl(false),
    isIndieOnlyChecked: new FormControl(false)
  });

  constructor(
    private gameService: GameService,
    private route: ActivatedRoute,
    private cdRef: ChangeDetectorRef
  ) {
    const savedTags = localStorage.getItem('selectedTags');
    if (savedTags) {
      this.selectedTags = new Set(JSON.parse(savedTags));
    }
    const savedTopTags = localStorage.getItem('topTagNames');
    if (savedTopTags) {
      this.topTagNames = (JSON.parse(savedTopTags));
    }
  }


  ngAfterViewInit(): void {
    this.isReady = true;
    this.cdRef.detectChanges();
  }

  ngOnInit(): void {
    this.filteredTags = this.topTagNames.slice(0, 5);

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

  filterTags(searchTerm: string): void { 
    if (!searchTerm) { 
      // If no search, revert to the initial 5 tags. 
      this.filteredTags = this.topTagNames.slice(0, 5); 
    } else { 
      // Filter tags loosely based on the search text (case-insensitive) and take the first five 
      this.filteredTags = this.topTagNames .filter(tag => tag.toLowerCase().includes(searchTerm.toLowerCase())) .slice(0, 5); } 
  }

  clearTagSearch(inputElement: HTMLInputElement): void {
    inputElement.value = '';
    this.filterTags('');
  }

  onCheckboxChange(tag: string, event: Event) {
    const input = event.target as HTMLInputElement;
    const checked = input?.checked ?? false;

    if (checked) {
      this.selectedTags.add(tag);
    } else {
      this.selectedTags.delete(tag);
    }
    console.log([...this.selectedTags]);

    // Save to localStorage
    localStorage.setItem('selectedTags', JSON.stringify([...this.selectedTags]));
  }
  
}
