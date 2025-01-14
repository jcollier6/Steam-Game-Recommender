# Steam Game Recommender

## Build

Run `python -m uvicorn main:app --reload` in the project folder to build it before starting the server.

## Development server

Run `ng serve` for a dev server. Navigate to `http://localhost:4200/`. 

## Notes

- Currently, you can enter any user's SteamId and it will provide recommendations based on their profile only if their profile is public. I plan on adding the ability for users to sign in to Steam so it'll work even when their account is private.

- The API Key is in a file named "environment.txt" that I've hidden from github. If you want to run this use your own Steam API key. Place the file in the main folder and only put the API key in the txt file, nothing else.
