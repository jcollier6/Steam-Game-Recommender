import { Component, Input, Output, EventEmitter, HostListener } from '@angular/core';

@Component({
  selector: 'app-custom-slider',
  standalone: true,
  templateUrl: './custom-slider.component.html',
  styleUrls: ['./custom-slider.component.css']
})
export class CustomSliderComponent {
  @Input() min: number = 0;
  @Input() max: number = 100;
  @Input() step: number = 1;
  @Input() value: number = 0;
  @Output() valueChange = new EventEmitter<number>();
  sliderWidth: number = 200; // Initial width, adjust as needed
  isDragging: boolean = false;
  handlePosition: number = 0;

  constructor() {
    this.updateHandlePosition();
  }


  @HostListener('window:mousemove', ['$event'])
  onMouseMove(event: MouseEvent) {
    if (this.isDragging) {
      this.updateValueFromMouse(event.clientX);
    }
  }

  @HostListener('window:mouseup')
  onMouseUp() {
    this.isDragging = false;
  }

  onMouseDown(event: MouseEvent) {
    this.isDragging = true;
    this.updateValueFromMouse(event.clientX);
  }

  updateValueFromMouse(clientX: number) {
     const sliderRect = document.querySelector('.slider-container')?.getBoundingClientRect();
    if (!sliderRect) return;

    let newPosition = clientX - sliderRect.left;
    if (newPosition < 0) {
      newPosition = 0;
    } else if (newPosition > sliderRect.width) {
      newPosition = sliderRect.width;
    }

    this.handlePosition = newPosition;
    this.value = this.calculateValueFromPosition(newPosition, sliderRect.width);
    this.valueChange.emit(this.value);
  }

  calculateValueFromPosition(position: number, totalWidth: number): number {
    const range = this.max - this.min;
    const value = this.min + (position / totalWidth) * range;
    return Math.round(value / this.step) * this.step;
  }

  updateHandlePosition() {
    const range = this.max - this.min;
    this.handlePosition = ((this.value - this.min) / range) * this.sliderWidth;
  }
}
