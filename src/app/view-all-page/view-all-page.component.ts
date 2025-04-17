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
  showMoreOrLess: string = 'Show more';
  tagDisplayLimit: number = 5;
  allTags: string[] = [];
  lastSearchTerm: string = '';
  noSearchableTags: boolean = false;

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
    
    this.gameService.getAllTags().subscribe((data) => {
      this.allTags = data;
    });
    this.gameService.getRecommendedGames().subscribe((data) => {
      this.allRecommendedGames = data;
    });
  }

  filterTags(searchTerm: string): void {
    this.lastSearchTerm = searchTerm; 
    if (!searchTerm) { 
      const checkedTags = Array.from(this.selectedTags);
      const orderedTags = new Set<string>();

      for (const tag of checkedTags) {
        orderedTags.add(tag);
      }
      for (const tag of this.topTagNames) {
        orderedTags.add(tag);
      }
      for (const tag of this.allTags) {
        orderedTags.add(tag);
      }
      this.filteredTags = Array.from(orderedTags).slice(0, this.tagDisplayLimit);
    } 
    else { 
      const search = searchTerm.toLowerCase();
      const startsWithMatches = this.allTags.filter(tag =>
        tag.toLowerCase().startsWith(search)
      );
      const includesMatches = this.allTags.filter(tag =>
        tag.toLowerCase().includes(search) && !tag.toLowerCase().startsWith(search)
      );
      const combined = Array.from(new Set([...startsWithMatches, ...includesMatches]));
      this.filteredTags = combined.slice(0, this.tagDisplayLimit); 

      if(this.filteredTags.length == 0){
        this.showMoreOrLess = 'No matching tags';
        this.noSearchableTags = true;
      }
    } 
    
    if(this.filteredTags.length != 0){
      this.noSearchableTags = false;
      if(this.tagDisplayLimit == 5){this.showMoreOrLess = 'Show more';}
      else{this.showMoreOrLess = 'Show less';}
    }
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

    localStorage.setItem('selectedTags', JSON.stringify([...this.selectedTags]));
  }

  toggleTagDisplayLimit(): void {
    this.tagDisplayLimit = this.tagDisplayLimit === 5 ? 15 : 5;
    this.showMoreOrLess = this.showMoreOrLess === 'Show more' ? 'Show less' : 'Show more';
    this.filterTags(this.lastSearchTerm)
  }
  
}
