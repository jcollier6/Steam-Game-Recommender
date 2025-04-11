import { Component, Input, Output, EventEmitter, ElementRef, ViewChild, AfterViewInit, HostListener, ChangeDetectorRef } from '@angular/core';
import { GameCardComponent } from "../game-card/game-card.component";
import { Game_Info } from '../../services/game.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-game-card-holder',
  standalone: true,
  imports: [ GameCardComponent, GameCardComponent, CommonModule ],
  templateUrl: './game-card-holder.component.html',
  styleUrl: './game-card-holder.component.css'
})
export class GameCardHolderComponent implements AfterViewInit {
  topBarWidth: string = '100%';
  fadeState: 'visible' | 'hidden' = 'visible';
  visibleGameList: Game_Info[] = [];
  gameListExist = false;
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

  private _resizeObserver!: ResizeObserver;
  private _resizeTimeout: any;
  private _maxCards: number = 0;
  private _gameList: Game_Info[] = [];

  @Output() viewAllClicked = new EventEmitter<string>();

  @Input() cardHolderTitle: string = '';

  @Input() isRecentTitle: boolean = false;

  @Input()
  set gameList(value: Game_Info[]) {
    this._gameList = value || [];
    this.gameListExist = this._gameList.length > 0;
  }
  
  get gameList(): Game_Info[] {
    return this._gameList;
  }
  
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

  constructor(private cdRef: ChangeDetectorRef) {}
  
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

  handleViewAllClick() {
    this.viewAllClicked.emit(this.cardHolderTitle)
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

  getWrappedSubsection(list: Game_Info[], start: number, count: number): Game_Info[] {
    const result: Game_Info[] = [];
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
