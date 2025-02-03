import { ChangeDetectionStrategy, Component, Input } from '@angular/core';

@Component({
  selector: 'app-navigation-bar',
  standalone: true,
  imports: [],
  templateUrl: './navigation-bar.component.html',
  styleUrl: './navigation-bar.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class NavigationBarComponent {
  @Input() signIn: string = "Log In";
}
