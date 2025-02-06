import { ChangeDetectionStrategy, Component, ChangeDetectorRef  } from '@angular/core';
import { Router, NavigationEnd } from '@angular/router';
import { filter } from 'rxjs/operators';

@Component({
  selector: 'app-navigation-bar',
  standalone: true,
  imports: [],
  templateUrl: './navigation-bar.component.html',
  styleUrl: './navigation-bar.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class NavigationBarComponent {
  signIn: string = 'Log In';

  constructor(
    private router: Router, 
    private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.router.events.pipe(
      filter((event): event is NavigationEnd => event instanceof NavigationEnd)
    ).subscribe(event => {
      if (event.url.includes('/home-page')) {
        this.signIn = 'Log Out';
      } else {
        this.signIn = 'Log In';
      }
      this.cdr.detectChanges();
    });
  }
}
