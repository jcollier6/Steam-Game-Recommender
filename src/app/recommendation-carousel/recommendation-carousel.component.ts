import { Component, ElementRef, ViewChild, Renderer2   } from '@angular/core';

@Component({
  selector: 'app-recommendation-carousel',
  standalone: true,
  imports: [],
  templateUrl: './recommendation-carousel.component.html',
  styleUrl: './recommendation-carousel.component.css'
})
export class RecommendationCarouselComponent {
  tags: string[] = ['Open World Survival Craft', 'PvE', 'Survival', 'Multiplayer', 'Co-op'];

  @ViewChild('tagsHolder', { static: true }) tagsHolder!: ElementRef<HTMLDivElement>;

  constructor(private renderer: Renderer2) {}

  ngAfterViewInit(): void {
    this.renderTags(this.tags);
  }

  renderTags(tags: string[]): void {
    const container = this.tagsHolder.nativeElement;
    container.innerHTML = ''; 

    tags.forEach(tagText => {
      const tag = this.renderer.createElement('span');
      this.renderer.addClass(tag, 'tag');
      const text = this.renderer.createText(tagText);
      this.renderer.appendChild(tag, text);
      this.renderer.appendChild(container, tag);     
      tag.style.fontSize = 'clamp(var(--font-size-small), 0.75vw, var(--font-size-standard))';
      tag.style.backgroundColor = 'var(--accent-color)';
      tag.style.borderRadius = 'var(--border-radius)';
      tag.style.padding = '0.2rem 0.5rem';
    });
  }
}
