import { Component, ElementRef, ViewChild, AfterViewInit, HostListener, ChangeDetectorRef } from '@angular/core';
import { GameCardComponent } from "../game-card/game-card.component";

@Component({
  selector: 'app-game-card-holder',
  standalone: true,
  imports: [ GameCardComponent, GameCardComponent ],
  templateUrl: './game-card-holder.component.html',
  styleUrl: './game-card-holder.component.css'
})
export class GameCardHolderComponent implements AfterViewInit {
  allCards = new Array(10);
  visibleCards: any[] = [];
  topBarWidth: string = "100%";

  private resizeObserver!: ResizeObserver;
  private resizeTimeout: any;

  constructor(private cdRef: ChangeDetectorRef) {}

  @ViewChild('container', { static: false }) containerRef!: ElementRef;
  ngAfterViewInit(): void {
    if (this.containerRef?.nativeElement) {
      this.resizeObserver = new ResizeObserver(() => {
        this.handleResize();
      });

      this.resizeObserver.observe(this.containerRef.nativeElement);

      // Initial call to handle first layout
      this.handleResize();
    }
  }

  private handleResize(): void {
    clearTimeout(this.resizeTimeout);
    this.resizeTimeout = setTimeout(() => {this.calculateVisibleCards();}, 0);
  }

  @HostListener('window:resize')
  onResize() {
    this.handleResize();
  }

  calculateVisibleCards() {
    console.log('Container width:', this.containerRef.nativeElement.offsetWidth);

    const containerWidth = this.containerRef.nativeElement.offsetWidth;
    const cardWidth = 208; // <-- adjust based on card width + margin (px count)
    const maxCards = Math.floor(containerWidth / cardWidth);
    const cardsWidth = cardWidth * maxCards
    this.topBarWidth = `${cardsWidth}px`;
    this.visibleCards = this.allCards.slice(0, maxCards);
    this.cdRef.detectChanges();
    console.log("max cards", maxCards)
  }

  ngOnDestroy(): void {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }

    clearTimeout(this.resizeTimeout);
  }
}
