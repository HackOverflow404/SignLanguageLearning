.PHONY: test

# Full test suite -- one command. Every test file under tests/ (extractor/model
# tests included; nothing is skipped by default). Run this after ANY change to
# extractor/, normalizer/, or features.py -- see CLAUDE.md.
test:
	uv run pytest tests/
