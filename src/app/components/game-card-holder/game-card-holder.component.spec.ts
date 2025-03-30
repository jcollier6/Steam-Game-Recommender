import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GameCardHolderComponent } from './game-card-holder.component';

describe('GameCardHolderComponent', () => {
  let component: GameCardHolderComponent;
  let fixture: ComponentFixture<GameCardHolderComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [GameCardHolderComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(GameCardHolderComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
