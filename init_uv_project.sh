#!/usr/bin/env bash
set -euo pipefail

# init_uv_project.sh
# Initialize a Python project managed by uv: create pyproject.toml, lockfile, and venv.

# Configure Python version here (e.g., "3.11", "3.12", "python3.11")
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

info() { printf "[INFO] %s\n" "$*"; }
warn() { printf "[WARN] %s\n" "$*"; }
err()  { printf "[ERROR] %s\n" "$*" >&2; }

# Detect project root as directory containing this script
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"


if ! command -v uv >/dev/null 2>&1; then
  err "uv command still not found after install. Ensure it's on your PATH and re-run."; exit 1
fi

# Ensure Python is available; prefer system python3 if no uv python configured
if ! command -v python3 >/dev/null 2>&1; then
  err "python3 not found. Please install Python 3.9+ and re-run."; exit 1
fi

# Create README.md if missing
if [ ! -f README.md ]; then
  info "Creating minimal README.md..."
  cat > README.md <<'README'
# homework_scientalab

Project description here.
README
fi

# Create src package structure if missing
if [ ! -d src/homework_scientalab ]; then
  info "Creating package structure at src/homework_scientalab/..."
  mkdir -p src/homework_scientalab
  touch src/homework_scientalab/__init__.py
fi

# Create pyproject.toml if missing
if [ ! -f pyproject.toml ]; then
  info "Creating minimal pyproject.toml for uv project..."
  cat > pyproject.toml <<'PYPROJECT'
[project]
name = "homework_scientalab"
version = "0.1.0"
description = ""
readme = "README.md"
requires-python = ">=3.11"
authors = [{ name = "Ludovic Arnould", email = "" }]
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/homework_scientalab"]

[dependency-groups]
dev = [
  "ruff>=0.6",
  "pytest>=8",
]
PYPROJECT
else
  info "Found existing pyproject.toml"
fi

# Create .venv managed by uv for project
info "Creating/updating virtual environment with uv (Python ${PYTHON_VERSION})..."
uv venv .venv --python "${PYTHON_VERSION}" --seed

# Sync dependencies (creates uv.lock)
info "Syncing dependencies and generating lockfile..."
uv sync

info "Project initialized. Activate venv with: source .venv/bin/activate"

