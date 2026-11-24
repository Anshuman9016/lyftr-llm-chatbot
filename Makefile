.PHONY: up down logs ps test dev-backend dev-frontend clean

up: .env
	docker compose up -d --build
	@echo ""
	@echo "Backend:  http://localhost:8000/health"
	@echo "Frontend: http://localhost:3000"

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

# Wipes the SQLite volume too.
clean:
	docker compose down -v

.env:
	@echo "No .env found - copying .env.example (put your real API keys in .env)"
	cp .env.example .env

test:
	cd backend && python -m pytest -q

dev-backend:
	cd backend && set -a && . ../.env && set +a && \
	DATABASE_URL=sqlite:///./app.db uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm install && npm run dev
