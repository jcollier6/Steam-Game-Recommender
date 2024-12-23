import { Component } from '@angular/core';
import { HttpClientModule, HttpClient } from '@angular/common/http';
import { NgFor } from '@angular/common';

interface RecommendedTag {
  tagid: number;
  name: string;
}

interface SteamUserData {
  rgRecommendedTags: RecommendedTag[]; // Recommended tags field
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [HttpClientModule, NgFor],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  recommendedTags: RecommendedTag[] = [];

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.get_recommended_tags();
  }

  get_recommended_tags() {
    console.log("Fetching recommended tags");
    this.http.get<SteamUserData>("https://store.steampowered.com/dynamicstore/userdata/?id=76561198293567287")
      .subscribe((res) => {
        this.recommendedTags = res.rgRecommendedTags ?? []; // Default to empty array if undefined
        console.log("Recommended Tags:", this.recommendedTags);
      }, error => {
        console.error("Error fetching data:", error);
      });
  }
}



  

  // references

  // get_tasks(){
  //   console.log("get")
  //   this.http.get(this.APIURL+"get_tasks").subscribe((res)=>{
  //     this.tasks=res;
  //   })
  // }

  // add_task(){
  //   console.log("here");
  //   let body=new FormData();
  //   body.append('task', this.newtask);
  //   this.http.post(this.APIURL+"add_task", body).subscribe((res)=>{
  //     alert(res)
  //     this.newtask="";
  //     this.get_tasks();
  //   })
  // }

  // delete_task(id:any){
  //   let body=new FormData();
  //   body.append('id', id);
  //   this.http.post(this.APIURL+"delete_task", body).subscribe((res)=>{
  //     alert(res)
  //     this.get_tasks();
  //   })
  // }
