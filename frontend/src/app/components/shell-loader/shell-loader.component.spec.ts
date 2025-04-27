import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ShellLoaderComponent } from './shell-loader.component';

describe('ShellLoaderComponent', () => {
  let component: ShellLoaderComponent;
  let fixture: ComponentFixture<ShellLoaderComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ShellLoaderComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ShellLoaderComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
