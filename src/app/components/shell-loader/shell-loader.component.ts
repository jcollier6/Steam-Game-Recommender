import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-shell-loader',
  standalone: true,
  imports: [],
  templateUrl: './shell-loader.component.html',
  styleUrl: './shell-loader.component.css'
})
export class ShellLoaderComponent {
  @Input() isReady = false;
}
