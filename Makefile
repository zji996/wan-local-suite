.PHONY: dev preprocess infer backend frontend

backend:
	uv run --python backend/app/main.py

frontend:
	cd frontend && npm run dev

dev:
	@echo "Starting frontend and backend dev servers"
	@$(MAKE) -j2 frontend backend

preprocess:
	./scripts/preprocess.sh $(job)

infer:
	./scripts/torchrun_animate.sh $(args)
