.PHONY: demo test

demo:
	./scripts/run_demo.sh

test:
	uv run pytest
