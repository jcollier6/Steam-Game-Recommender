import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BottomInfoBarComponent } from './bottom-info-bar.component';

describe('BottomInfoBarComponent', () => {
  let component: BottomInfoBarComponent;
  let fixture: ComponentFixture<BottomInfoBarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BottomInfoBarComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(BottomInfoBarComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
