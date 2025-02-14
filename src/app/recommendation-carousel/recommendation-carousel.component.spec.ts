import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RecommendationCarouselComponent } from './recommendation-carousel.component';

describe('RecommendationCarouselComponent', () => {
  let component: RecommendationCarouselComponent;
  let fixture: ComponentFixture<RecommendationCarouselComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RecommendationCarouselComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(RecommendationCarouselComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
