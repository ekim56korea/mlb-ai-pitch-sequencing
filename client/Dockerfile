# 🟢 중요: Node 18이 아닌 20을 사용합니다.
FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json ./

# 충돌 방지를 위해 legacy-peer-deps 옵션 사용
RUN npm install --legacy-peer-deps

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]