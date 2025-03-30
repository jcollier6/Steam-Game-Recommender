import { Component, ElementRef, ViewChild, AfterViewInit, HostListener, ChangeDetectorRef } from '@angular/core';
import { GameCardComponent } from "../game-card/game-card.component";
import { Recommended_Game, GameService } from '../../services/game.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-game-card-holder',
  standalone: true,
  imports: [ GameCardComponent, GameCardComponent, CommonModule ],
  templateUrl: './game-card-holder.component.html',
  styleUrl: './game-card-holder.component.css'
})
export class GameCardHolderComponent implements AfterViewInit {
  cardHolderTitle: string = '';
  topBarWidth: string = '100%';
  fadeState: 'visible' | 'hidden' = 'visible';
  gameList: Recommended_Game[] = [];
  visibleGameList: Recommended_Game[] = [];
  gameListExist = false;
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

  private _resizeObserver!: ResizeObserver;
  private _resizeTimeout: any;
  private _maxCards: number = 0;

  get maxCards(): number {
    return this._maxCards;
  }

  set maxCards(value: number) {
    if (value !== this._maxCards) {
      this._maxCards = value;
      if (this.gameListExist) {
        this.updateGameList('');
      }
    }
  }

  constructor(
    private cdRef: ChangeDetectorRef,
    private gameService: GameService
  ) {}

  ngOnInit(): void {
    this.gameService.getRecommendedGames().subscribe((data) => {
      this.gameList = data;
      if (this.gameList.length > 0) {
        this.gameListExist = true;
        this.currentIndex = 0;
      }
      else {
        this.gameListExist = false;
      }
    });
  }
  
  @ViewChild('container', { static: false }) containerRef!: ElementRef;
  ngAfterViewInit(): void {
    if (this.containerRef?.nativeElement) {
      this._resizeObserver = new ResizeObserver(() => {
        this.handleResize();
      });

      this._resizeObserver.observe(this.containerRef.nativeElement);
      // Initial call to handle first layout
      this.handleResize();
      
    }
  }

  private handleResize(): void {
    clearTimeout(this._resizeTimeout);
    this._resizeTimeout = setTimeout(() => {this.calculatevisibleCardCount();}, 10);
  }

  @HostListener('window:resize')
  onResize() {
    this.handleResize();
  }

  calculatevisibleCardCount() {
    const containerWidth = this.containerRef.nativeElement.offsetWidth;
    const cardWidth = 208; // <-- adjust based on card width + margin + arrow widths (px count)
    this.maxCards = Math.floor(containerWidth / cardWidth);
    const cardsListWidth = cardWidth * this.maxCards
    this.topBarWidth = `${cardsListWidth}px`;
    this.cdRef.detectChanges();
  }

  updateGameList(direction: 'next' | 'previous' | ''): void {
    if (direction === 'next') {
      this.currentIndex = (this.currentIndex + 1) % this.gameList.length;
    } else if (direction === 'previous') {
      this.currentIndex = (this.currentIndex - 1 + this.gameList.length) % this.gameList.length;
    }
    this.visibleGameList = this.getWrappedSubsection(this.gameList, this.currentIndex, this.maxCards);
    this.cdRef.detectChanges();
  }

  getWrappedSubsection(list: Recommended_Game[], start: number, count: number): Recommended_Game[] {
    const result: Recommended_Game[] = [];
    const length = list.length;
  
    for (let i = 0; i < count; i++) {
      const index = (start + i) % length;
      result.push(list[index]);
    }
    return result;
  }

  ngOnDestroy(): void {
    if (this._resizeObserver) {
      this._resizeObserver.disconnect();
    }

    clearTimeout(this._resizeTimeout);
  }
}
