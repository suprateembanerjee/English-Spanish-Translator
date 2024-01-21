# Author: Suprateem Banerjee [www.github.com/suprateembanerjee]

FROM node:18-alpine

WORKDIR /app

COPY package.json package-lock.json .

RUN npm install

COPY . .

EXPOSE 5173

CMD [ "npm", "dev", "run", "preview"]