# Steam Game Recommender

## Build

Run `python -m uvicorn main:app --reload` in the project folder to build it before starting the server.

## Development server

Run `ng serve` for a dev server. Navigate to `http://localhost:4200/`. The application will automatically reload if you change any of the source files.

## Notes

-Currently, it is running the model based on a single random user's information that I manually gave it. I plan on adding the ability for users to sign in to Steam for the model to use their information instead.

-API Key is in a file named "environment.txt" that I've hidden from github. If you want to run this use your own Steam API key. Place the file in the main folder and only put the api key in the txt file, nothing else.