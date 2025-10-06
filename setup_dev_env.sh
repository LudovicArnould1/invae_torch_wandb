#!/usr/bin/env bash
set -euo pipefail

# setup_dev_env.sh
# Bootstrap a developer machine to work on this project using uv.

info() { printf "[INFO] %s\n" "$*"; }
warn() { printf "[WARN] %s\n" "$*"; }
err()  { printf "[ERROR] %s\n" "$*" >&2; }

# Detect project root as directory containing this script
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

info "Bootstrapping development environment for project at: $PROJECT_ROOT"

# Dependencies: curl or wget, python3, uv
if ! command -v uv >/dev/null 2>&1; then
  warn "uv not found. Attempting to install via official installer..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    err "Neither curl nor wget available to install uv. Install uv and re-run."; exit 1
  fi
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  err "uv command still not found after install. Ensure it's on your PATH and re-run."; exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  err "python3 not found. Please install Python 3.9+ and re-run."; exit 1
fi

# Initialize project if needed
if [ ! -f pyproject.toml ]; then
  warn "pyproject.toml not found. Running init script to create a minimal project..."
  bash "$PROJECT_ROOT/init_uv_project.sh"
fi

# Create venv and sync deps
info "Setting up virtual environment and syncing dependencies..."
uv venv --python "python3" --seed .venv
uv sync

# Offer to install pre-commit hooks if config exists
if [ -f .pre-commit-config.yaml ]; then
  if command -v pre-commit >/dev/null 2>&1; then
    info "Installing pre-commit hooks..."
    pre-commit install || warn "pre-commit install failed"
  else
    warn "pre-commit not found. Consider installing with 'uv add --dev pre-commit'"
  fi
fi

info "Done. Activate venv with: source .venv/bin/activate"
info "To run tests: uv run pytest -q"

