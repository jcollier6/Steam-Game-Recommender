# Stage 1: Build Angular app
FROM node:18 AS build

WORKDIR /app
COPY . .
RUN npm install --omit=optional
RUN npm run build --prod

# Stage 2: Serve with Nginx
FROM nginx:alpine

RUN rm -rf /usr/share/nginx/html/*

COPY --from=build /app/dist/my-app/browser /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
