import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EnterSteamIdComponent } from './enter-steam-id.component';

describe('EnterSteamIdComponent', () => {
  let component: EnterSteamIdComponent;
  let fixture: ComponentFixture<EnterSteamIdComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EnterSteamIdComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(EnterSteamIdComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
