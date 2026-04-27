#!/usr/bin/env bash
# Creates a local virtual environment, activates it, and installs all dependencies
# (runtime + dev/testing).
# Usage:  bash setup_venv.sh

set -euo pipefail

VENV_DIR=".venv"

echo "==> Creating virtual environment in ${VENV_DIR}/ ..."
python3 -m venv "${VENV_DIR}"

echo "==> Activating ..."
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip ..."
pip install --upgrade pip --quiet

echo "==> Installing runtime + dev dependencies ..."
pip install -r requirements.txt

echo ""
echo "Done! Runtime and dev dependencies (pytest, ipywidgets, mlflow, plotly) installed."
echo "To activate the environment in a new shell run:"
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "To run tests:"
echo "    pytest tests/"
