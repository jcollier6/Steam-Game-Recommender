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
  steamTagCounts: Record<string,number> = {};

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
    this.gameService.getSteamTagCounts().subscribe(counts => {
      this.steamTagCounts = counts;
    });
  }

  filterTags(searchTerm: string): void {
  this.lastSearchTerm = searchTerm;
  const limit       = this.tagDisplayLimit;
  const searchLower = searchTerm.toLowerCase();

  // 1) Pool = allTags matching the search, or allTags if empty
  const pool = searchTerm
    ? this.allTags.filter(t => t.toLowerCase().includes(searchLower))
    : this.allTags;

  const result: string[] = [];
  const seen = new Set<string>();

  // 2) Checked tags first—but only those matching the search
  for (const tag of this.selectedTags) {
    if (result.length >= limit) break;
    if (searchTerm && !tag.toLowerCase().includes(searchLower)) {
      continue;
    }
    if (!seen.has(tag)) {
      result.push(tag);
      seen.add(tag);
    }
  }

  // 3) Then topTagNames (only if in pool)
  for (const tag of this.topTagNames) {
    if (result.length >= limit) break;
    if (!seen.has(tag) && pool.includes(tag)) {
      result.push(tag);
      seen.add(tag);
    }
  }

  // 4) Fill the rest up to limit by highest steamTagCounts
  if (result.length < limit) {
    const needed = limit - result.length;
    const topK: string[] = [];

    for (const tag of pool) {
      if (seen.has(tag)) continue;
      const cnt = this.steamTagCounts[tag] || 0;
      let pos = 0;
      while (pos < topK.length &&
             (this.steamTagCounts[topK[pos]] || 0) >= cnt) {
        pos++;
      }
      if (pos < needed) {
        topK.splice(pos, 0, tag);
        if (topK.length > needed) topK.length = needed;
      }
    }

    result.push(...topK);
  }

  this.filteredTags = result;

  // 5) Update UI flags
  this.noSearchableTags = result.length === 0;
  if (!this.noSearchableTags) {
    this.showMoreOrLess =
      this.tagDisplayLimit === result.length ? 'Show more' : 'Show less';
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

  getTagCount(tag: string): number {
    return this.steamTagCounts[tag] ?? 0;
  }
  
  excludeTag(tag: string): void {
    return
  }
}
